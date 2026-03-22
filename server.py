"""
Minimal API: GET /health and POST /v1/analyze (image → screening JSON).

Run:
  uvicorn server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vision_model: str = "gemini-2.5-flash"
    gemini_api_key: Optional[str] = None
    screening_api_key: Optional[str] = None
    cors_origins: str = "*"
    max_upload_mb: int = 10

    def provider_api_key(self) -> Optional[str]:
        return self.gemini_api_key or self.screening_api_key


settings = Settings()


def _extract_json_obj(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    obj = re.search(r"(\{.*\})", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(1))
        except Exception:
            pass
    return None


def run_screening(
    image_bytes: bytes,
    mime_type: str,
    *,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    if genai is None or genai_types is None:
        return {"ok": False, "error": "Vision client library not installed."}

    prompt = (
        "You assist with glaucoma-related eye screening from ONE image. "
        "First set is_fundus_image true only for a clear retinal fundus photo showing the optic disc. "
        "\n"
        "You MUST include glaucoma_status, one of: yes | no | cannot_determine. "
        "This is screening language only, not a definitive diagnosis. "
        "- FUNDUS with a clearly visible optic disc and findings suggestive of glaucoma "
        "(e.g. large cup-to-disc, rim thinning): glaucoma_status=yes, align severity with risk. "
        "- FUNDUS with a clearly visible disc that appears within normal limits for cupping: "
        "glaucoma_status=no, severity typically normal or mild. "
        "- FUNDUS but disc obscured, too blurry, or edge of field only: glaucoma_status=cannot_determine. "
        "\n"
        "NON-FUNDUS (is_fundus_image=false), anterior / external / phone photo rules: "
        "glaucoma_status=yes is ONLY for fundus views; do not set yes from external-only photos. "
        "- If the anterior segment looks largely NORMAL and image quality is adequate (no major opacity, "
        "no severe injection, no acute emergency appearance): use glaucoma_status=no. "
        "Meaning: no visible signs in THIS photo that suggest glaucoma-related emergencies or obvious "
        "nerve-related clues from the front of the eye; it is NOT proof that the person does not have "
        "glaucoma (optic nerve not seen). Say that clearly in reason. "
        "- If there is SIGNIFICANT pathology (e.g. severe corneal clouding, marked injection, "
        "pterygium obscuring view, or other findings that block assessment): glaucoma_status=cannot_determine "
        "and severity reflects anterior urgency, not glaucoma stage. "
        "- If the image is too blurry, off-angle, or ambiguous: glaucoma_status=cannot_determine. "
        "Severity on non-fundus images describes visible anterior-segment urgency only, NOT glaucoma stage. "
        "\n"
        "A) FUNDUS (is_fundus_image=true): Estimate glaucoma risk from disc/cup appearance. "
        "Use severity normal|mild|moderate|severe|uncertain based on visible CDR/disc cues; "
        "uncertain only if the disc is not visible or quality is too poor (and then glaucoma_status=cannot_determine). "
        "\n"
        "B) EXTERNAL / ANTERIOR PHOTO (is_fundus_image=false): "
        "Normal-appearing eye → glaucoma_status=no as above; severity=normal; reason mentions optic nerve not examined. "
        "Concerning anterior findings → severity mild|moderate|severe for urgency; glaucoma_status=cannot_determine. "
        "\n"
        "Return STRICT JSON only with keys: "
        "glaucoma_status (yes|no|cannot_determine), "
        "severity (normal|mild|moderate|severe|uncertain), "
        "confidence (float 0 to 1), "
        "reason (short sentence), "
        "is_fundus_image (boolean), "
        "warning (string, optional), "
        "follow_up (array of short strings), "
        "food_to_take (array of short strings), "
        "lifestyle_habits (array of short strings). "
        "Always provide 2 to 5 short items in EACH of food_to_take and lifestyle_habits "
        "(general eye-friendly diet ideas, exercise, sleep, UV protection, not smoking, "
        "taking prescribed eye drops as directed if applicable). "
        "Even for severe or urgent cases, include these as supportive adjuncts—state in warning "
        "or follow_up that they do not replace urgent specialist care. "
        "Do not claim a medical diagnosis; triage-style guidance only."
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
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
            return {
                "ok": False,
                "error": "Model returned non-JSON output.",
                "raw_response": raw_text[:2000],
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    gs = payload.get("glaucoma_status")
    if gs not in ("yes", "no", "cannot_determine"):
        payload["glaucoma_status"] = "cannot_determine"

    payload["ok"] = True
    return payload


app = FastAPI(
    title="Eye screening API",
    description="Health check and image-based screening analysis.",
    version="1.0.0",
)

_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str


class ImageAnalysisResponse(BaseModel):
    """Structured output from the vision model (extra fields are preserved)."""

    model_config = ConfigDict(extra="allow")

    ok: bool = Field(description="True when the model returned usable JSON.")
    glaucoma_status: Optional[str] = Field(
        None,
        description=(
            "yes = fundus suggests glaucoma risk. "
            "no = fundus disc looks low-risk OR normal-appearing anterior photo with no visible "
            "glaucoma-related red flags in this view (optic nerve still not examined). "
            "cannot_determine = disc not assessable, significant anterior pathology, or poor image. "
            "Not a definitive diagnosis."
        ),
    )
    severity: Optional[str] = Field(
        None,
        description=(
            "For fundus: risk level from disc cues. "
            "For non-fundus: visible anterior urgency only, not glaucoma stage."
        ),
    )
    confidence: Optional[float] = Field(
        None, description="Model self-reported confidence, 0–1."
    )
    reason: Optional[str] = Field(None, description="Short explanation.")
    is_fundus_image: Optional[bool] = Field(
        None, description="Whether the image looks like a retinal fundus photo."
    )
    warning: Optional[str] = None
    follow_up: Optional[list[str]] = None
    food_to_take: Optional[list[str]] = None
    lifestyle_habits: Optional[list[str]] = None
    error: Optional[str] = Field(None, description="Present when ok is false.")
    raw_response: Optional[str] = Field(
        None, description="Truncated raw text if JSON parsing failed."
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


async def _analyze_image_upload(file: UploadFile) -> JSONResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image, got {file.content_type}",
        )

    raw = await file.read()
    max_b = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_b:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.max_upload_mb} MB)",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    key = settings.provider_api_key()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="Set GEMINI_API_KEY or SCREENING_API_KEY in the server environment.",
        )

    mime = file.content_type or "image/jpeg"
    result = run_screening(
        raw,
        mime,
        api_key=key,
        model=settings.vision_model,
    )
    body = ImageAnalysisResponse.model_validate(result)
    status_code = 200 if result.get("ok") else 502
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
    )


@app.post(
    "/v1/analyze",
    response_model=ImageAnalysisResponse,
    summary="Analyze image",
    description=(
        "Upload a fundus or eye photo. Returns structured screening-style JSON. "
        "Not a medical diagnosis."
    ),
    tags=["Analysis"],
    responses={
        502: {
            "description": "Upstream model error or invalid response",
            "model": ImageAnalysisResponse,
        },
    },
)
async def analyze_image(
    file: UploadFile = File(
        ...,
        description="Image file (e.g. JPEG, PNG, WebP).",
    ),
):
    return await _analyze_image_upload(file)


@app.post(
    "/v1/screen",
    include_in_schema=False,
)
async def analyze_image_legacy(file: UploadFile = File(...)):
    """Same as POST /v1/analyze; kept for older clients."""
    return await _analyze_image_upload(file)
