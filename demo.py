from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


@dataclass
class GlaucomaConfig:
    """
    Configuration for glaucoma detection from a single eye (fundus) image.
    """

    image_size: Tuple[int, int] = (256, 256)
    device: str = "cpu"
    weights_path: Optional[Path] = None
    prob_threshold: float = 0.5
    min_disc_diameter_px: float = 8.0
    min_cup_diameter_px: float = 4.0
    gemini_model: str = "gemini-2.5-flash"


class SimpleUNet(nn.Module):
    """
    Minimal U-Net–style model producing 2-channel segmentation:
    - channel 0: optic disc mask
    - channel 1: optic cup mask
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.down1(x)
        p1 = self.pool1(e1)
        e2 = self.down2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        return out


class GlaucomaDetector:
    """
    Wraps preprocessing, model inference, CDR computation, and severity grading.
    """

    def __init__(self, cfg: Optional[GlaucomaConfig] = None):
        self.cfg = cfg or GlaucomaConfig()
        self.device = torch.device(self.cfg.device)
        self.model = SimpleUNet().to(self.device)
        self.using_trained_weights = False

        if self.cfg.weights_path is not None and self.cfg.weights_path.is_file():
            state = torch.load(self.cfg.weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.using_trained_weights = True

        # Keep model in inference mode for this script.
        self.model.eval()

    def _preprocess_bgr(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert BGR uint8 image to model input tensor.
        Returns tensor (1, 1, H, W) and original size (W, H).
        """
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty or invalid image array.")

        original_size = (img_bgr.shape[1], img_bgr.shape[0])  # (W, H)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, self.cfg.image_size, interpolation=cv2.INTER_AREA
        )
        img = resized.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img).to(self.device)
        return tensor, original_size

    def _preprocess(self, image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Load image, convert to grayscale, resize, normalize.
        Returns tensor (1, 1, H, W) and original size.
        """
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self._preprocess_bgr(img_bgr)

    def analyze_image_bytes(self, data: bytes) -> Dict[str, float | str | bool]:
        """
        Same as analyze_image but decodes an in-memory image (e.g. uploaded bytes).
        """
        buf = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image bytes (unsupported or corrupt).")

        x, _ = self._preprocess_bgr(img_bgr)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        disc_prob = probs[0]
        cup_prob = probs[1]
        disc_mask = (disc_prob > self.cfg.prob_threshold).astype(np.uint8)
        cup_mask = (cup_prob > self.cfg.prob_threshold).astype(np.uint8)

        disc_diam = self._mask_diameter(disc_mask)
        cup_diam = self._mask_diameter(cup_mask)
        if disc_diam <= 0:
            cdr = 0.0
        else:
            cdr = float(cup_diam / disc_diam)

        likely_invalid_segmentation = (
            disc_diam < self.cfg.min_disc_diameter_px
            or cup_diam < self.cfg.min_cup_diameter_px
        )
        severity = self._grade_cdr(cdr)

        return {
            "cdr": cdr,
            "severity": severity,
            "cup_diameter_px": cup_diam,
            "disc_diameter_px": disc_diam,
            "likely_invalid_segmentation": likely_invalid_segmentation,
            "used_trained_weights": self.using_trained_weights,
        }

    @staticmethod
    def _mask_diameter(mask: np.ndarray) -> float:
        """
        Approximate diameter of mask region using bounding box diagonal.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0.0

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        width = float(x_max - x_min + 1)
        height = float(y_max - y_min + 1)

        # Use max dimension as "diameter"
        return max(width, height)

    @staticmethod
    def _grade_cdr(cdr: float) -> str:
        """
        Map CDR value to glaucoma severity levels.
        """
        if cdr < 0.4:
            return "normal"
        elif 0.4 <= cdr < 0.6:
            return "mild"
        elif 0.6 <= cdr < 0.7:
            return "moderate"
        else:
            return "severe"

    @staticmethod
    def _risk_level_to_cdr(risk_level: str) -> float:
        """
        Convert external-eye Gemini risk level to an estimated CDR-like score
        for unified CLI output. This is a heuristic mapping.
        """
        mapping = {
            "low": 0.35,
            "medium": 0.55,
            "high": 0.75,
            "uncertain": 0.50,
        }
        return mapping.get(str(risk_level).lower(), 0.50)

    def _gemini_second_opinion(
        self,
        image_path: Path,
        primary_result: Optional[Dict[str, float | str | bool]],
        api_key: str,
    ) -> Dict[str, Any]:
        """
        Ask Gemini for an optional second-opinion screening signal.
        """
        if genai is None or genai_types is None:
            return {
                "ok": False,
                "error": "google-genai is not installed. Run: pip install google-genai",
            }

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/jpeg"

        try:
            image_bytes = image_path.read_bytes()
        except Exception as exc:  # pragma: no cover - defensive I/O branch
            return {"ok": False, "error": f"Could not read image for Gemini: {exc}"}

        if primary_result is None:
            context_text = (
                "No local segmentation output is available. "
                "Infer from the image only."
            )
        else:
            context_text = (
                "Use this local segmentation output as supporting signal: "
                f"cdr={primary_result['cdr']:.3f}, "
                f"disc_diameter_px={primary_result['disc_diameter_px']:.1f}, "
                f"cup_diameter_px={primary_result['cup_diameter_px']:.1f}, "
                f"local_severity={primary_result['severity']}."
            )

        prompt = (
            "You are assisting with glaucoma screening from an eye image. "
            "First decide whether this is a retinal fundus image. "
            "If it is not fundus, set severity='uncertain'. "
            "Return STRICT JSON only with keys: "
            "severity (normal|mild|moderate|severe|uncertain), "
            "confidence (float 0 to 1), "
            "reason (short sentence), "
            "is_fundus_image (boolean), "
            "warning (string, optional), "
            "follow_up (array of short strings), "
            "food_to_take (array of short strings), "
            "lifestyle_habits (array of short strings). "
            + context_text
        )

        def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
            text = text.strip()
            if not text:
                return None
            # 1) Raw JSON
            try:
                return json.loads(text)
            except Exception:
                pass
            # 2) Markdown fenced JSON
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if fenced:
                try:
                    return json.loads(fenced.group(1))
                except Exception:
                    pass
            # 3) First JSON object in text
            obj = re.search(r"(\{.*\})", text, re.DOTALL)
            if obj:
                try:
                    return json.loads(obj.group(1))
                except Exception:
                    pass
            return None

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.cfg.gemini_model,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
                contents=[
                    genai_types.Part.from_text(text=prompt),
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
            raw_text = (response.text or "").strip()
            payload = _extract_json_obj(raw_text)
            if payload is None:
                raise ValueError("Gemini returned non-JSON output.")
        except ValueError:
            return {
                "ok": False,
                "error": "Gemini returned non-JSON output.",
                "raw_response": (response.text if "response" in locals() else ""),
            }
        except Exception as exc:
            return {"ok": False, "error": f"Gemini request failed: {exc}"}

        payload["ok"] = True
        return payload

    def _gemini_external_eye_screen(self, image_path: Path, api_key: str) -> Dict[str, Any]:
        """
        Gemini-only screening for non-fundus (anterior/external eye) photos.
        Not a diagnosis; only visible-sign risk triage.
        """
        if genai is None or genai_types is None:
            return {
                "ok": False,
                "error": "google-genai is not installed. Run: pip install google-genai",
            }

        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/jpeg"

        try:
            image_bytes = image_path.read_bytes()
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": f"Could not read image for Gemini: {exc}"}

        prompt = (
            "This is an external/anterior eye photo screening task (not fundus CDR). "
            "Return STRICT JSON only with keys: "
            "risk_level (low|medium|high|uncertain), "
            "confidence (float 0 to 1), "
            "visible_findings (array of short strings), "
            "reason (short sentence), "
            "recommendation (short sentence), "
            "warning (string), "
            "follow_up (array of short strings), "
            "food_to_take (array of short strings), "
            "lifestyle_habits (array of short strings). "
            "If image quality is poor or findings are not reliable, set risk_level='uncertain'. "
            "Do not claim diagnosis; provide triage-style guidance only."
        )

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.cfg.gemini_model,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
                contents=[
                    genai_types.Part.from_text(text=prompt),
                    genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
            raw_text = (response.text or "").strip()
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": "Gemini returned non-JSON output.",
                "raw_response": (response.text if "response" in locals() else ""),
            }
        except Exception as exc:
            return {"ok": False, "error": f"Gemini request failed: {exc}"}

        payload["ok"] = True
        return payload

    def analyze_image(self, image_path: str | Path) -> Dict[str, float | str | bool]:
        """
        End-to-end processing for a single image.

        Steps:
        - Preprocessing: grayscale, resize, tensor normalization.
        - Neural net: cup & disc segmentation.
        - CDR: cup-to-disc diameter ratio.
        - Severity: normal / mild / moderate / severe.
        """
        data = Path(image_path).read_bytes()
        return self.analyze_image_bytes(data)


def run_gemini_pipeline(
    detector: GlaucomaDetector,
    image_path: Path,
    primary_result: Optional[Dict[str, Any]],
    api_key: str,
    image_type: str,
) -> Dict[str, Any]:
    """
    Run Gemini second-opinion or external screening with optional auto modality switch.
    image_type: auto | fundus | external (fundus uses second-opinion path like auto without external-only).
    """
    if image_type == "external":
        return detector._gemini_external_eye_screen(
            image_path=image_path, api_key=api_key
        )
    gemini_result = detector._gemini_second_opinion(
        image_path=image_path,
        primary_result=primary_result,
        api_key=api_key,
    )
    if (
        image_type == "auto"
        and gemini_result.get("ok")
        and gemini_result.get("is_fundus_image") is False
    ):
        external_result = detector._gemini_external_eye_screen(
            image_path=image_path,
            api_key=api_key,
        )
        if external_result.get("ok"):
            gemini_result = {
                "ok": True,
                "mode": "external",
                "fundus_check": gemini_result,
                "external_screen": external_result,
            }
        else:
            gemini_result = {
                "ok": True,
                "mode": "fundus_non_match",
                "fundus_check": gemini_result,
                "external_screen_error": external_result.get("error"),
            }
    return gemini_result


def interpret_gemini_for_final(
    detector: GlaucomaDetector,
    gemini_result: Dict[str, Any],
    image_type: str,
    local_result: Optional[Dict[str, Any]],
) -> Tuple[float, str, str, list[str], list[str], list[str]]:
    """
    Map Gemini output + local CDR to unified final_cdr, severity, and guidance fields.
    """
    final_cdr: Optional[float] = None
    if local_result is not None:
        final_cdr = float(local_result["cdr"])
    final_recommendation = ""
    final_follow_up: list[str] = []
    final_food: list[str] = []
    final_habits: list[str] = []

    if not gemini_result.get("ok"):
        pass
    elif gemini_result.get("mode") == "external":
        ext = gemini_result.get("external_screen", {})
        final_cdr = detector._risk_level_to_cdr(ext.get("risk_level", "uncertain"))
        final_recommendation = (
            ext.get("recommendation")
            or ext.get("reason")
            or "Please consult an eye care professional for full evaluation."
        )
        final_follow_up = list(ext.get("follow_up", []) or [])
        final_food = list(ext.get("food_to_take", []) or [])
        final_habits = list(ext.get("lifestyle_habits", []) or [])
    elif image_type == "external":
        final_cdr = detector._risk_level_to_cdr(
            gemini_result.get("risk_level", "uncertain")
        )
        final_recommendation = (
            gemini_result.get("recommendation")
            or gemini_result.get("reason")
            or "Please consult an eye care professional for full evaluation."
        )
        final_follow_up = list(gemini_result.get("follow_up", []) or [])
        final_food = list(gemini_result.get("food_to_take", []) or [])
        final_habits = list(gemini_result.get("lifestyle_habits", []) or [])
    else:
        if final_cdr is None:
            sev = str(gemini_result.get("severity", "uncertain")).lower()
            if sev == "normal":
                final_cdr = 0.35
            elif sev == "mild":
                final_cdr = 0.50
            elif sev == "moderate":
                final_cdr = 0.65
            elif sev == "severe":
                final_cdr = 0.75
            else:
                final_cdr = 0.50
        final_recommendation = (
            gemini_result.get("warning")
            or gemini_result.get("reason")
            or "Please consult an eye care professional for full evaluation."
        )
        final_follow_up = list(gemini_result.get("follow_up", []) or [])
        final_food = list(gemini_result.get("food_to_take", []) or [])
        final_habits = list(gemini_result.get("lifestyle_habits", []) or [])

    if final_cdr is None:
        final_cdr = 0.0
    final_severity = detector._grade_cdr(final_cdr)
    return (
        final_cdr,
        final_severity,
        final_recommendation,
        final_follow_up,
        final_food,
        final_habits,
    )


def run_cli():
    # Load local .env (same folder as this script) when python-dotenv is installed.
    if load_dotenv is not None:
        load_dotenv(Path(__file__).with_name(".env"))

    parser = argparse.ArgumentParser(
        description="Glaucoma detection using CDR (cup-to-disc ratio) from an eye image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input eye (fundus) image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to trained model weights (.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to run on, e.g. "cpu" or "cuda".',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to binarize disc/cup masks.",
    )
    parser.add_argument(
        "--gemini-confirm",
        action="store_true",
        help="Ask Gemini for a second-opinion severity estimate.",
    )
    parser.add_argument(
        "--gemini-only",
        action="store_true",
        help="Use Gemini only (skip local segmentation model).",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model for second-opinion check.",
    )
    parser.add_argument(
        "--image-type",
        type=str,
        default="auto",
        choices=["auto", "fundus", "external"],
        help="Image modality handling for Gemini: auto, fundus, or external.",
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1 (exclusive).")

    cfg = GlaucomaConfig(
        device=args.device,
        weights_path=Path(args.weights) if args.weights else None,
        prob_threshold=args.threshold,
        gemini_model=args.gemini_model,
    )
    detector = GlaucomaDetector(cfg)
    image_path = Path(args.image)
    result = None

    if not args.gemini_only:
        result = detector.analyze_image(image_path)

    print(f"Image: {args.image}")
    final_cdr: Optional[float] = None
    final_recommendation = ""
    final_follow_up: list[str] = []
    final_food: list[str] = []
    final_habits: list[str] = []

    if args.gemini_only:
        print("Local model: skipped (--gemini-only enabled)")
    else:
        print(
            "Weights used: "
            + (
                str(cfg.weights_path)
                if result["used_trained_weights"]
                else "None (random model weights)"
            )
        )
        print(f"CDR (cup-to-disc ratio): {result['cdr']:.3f}")
        print(f"Severity level: {result['severity']}")
        print(f"Disc diameter (px): {result['disc_diameter_px']:.1f}")
        print(f"Cup diameter (px):  {result['cup_diameter_px']:.1f}")
        final_cdr = float(result["cdr"])

        if not result["used_trained_weights"]:
            print(
                "WARNING: No trained weights loaded. Predictions are not clinically meaningful."
            )

        if result["likely_invalid_segmentation"]:
            print(
                "WARNING: Segmentation looks unreliable (very small disc/cup). "
                "Use a retinal fundus image and trained weights."
            )

    if args.gemini_confirm:
        gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print(
                "Gemini second opinion skipped: set --gemini-api-key "
                "or GEMINI_API_KEY environment variable."
            )
            return

        gemini_result = run_gemini_pipeline(
            detector,
            image_path,
            result,
            gemini_api_key,
            args.image_type,
        )
        print("\nGemini second opinion:")
        if not gemini_result.get("ok"):
            print("- status: failed")
            print(f"- reason: {gemini_result.get('error', 'unknown error')}")
        (
            final_cdr,
            final_severity,
            final_recommendation,
            final_follow_up,
            final_food,
            final_habits,
        ) = interpret_gemini_for_final(
            detector, gemini_result, args.image_type, result
        )
    else:
        if final_cdr is None:
            final_cdr = 0.0
        final_severity = detector._grade_cdr(final_cdr)

    # Unified short output requested by user.
    print("\nFinal result:")
    print(f"CDR: {final_cdr:.3f}")
    print(f"Severity level: {final_severity}")
    if final_recommendation:
        print(f"Recommendation: {final_recommendation}")
    if final_follow_up or final_food or final_habits:
        print("Follow-up:")
        for item in final_follow_up:
            print(f"- {item}")
        print("Food to take:")
        for item in final_food:
            print(f"- {item}")
        print("Lifestyle:")
        for item in final_habits:
            print(f"- {item}")
    else:
        print(
            "Guidance unavailable from Gemini. Re-run with --gemini-confirm "
            "to get food and lifestyle suggestions."
        )


if __name__ == "__main__":
    run_cli()