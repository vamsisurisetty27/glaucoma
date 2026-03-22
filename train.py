from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from demo import SimpleUNet


@dataclass
class TrainConfig:
    data_dir: Path
    output_path: Path
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    val_split: float = 0.2
    device: str = "cpu"
    seed: int = 42


class FundusSegmentationDataset(Dataset):
    """
    Expected dataset layout:
      data_dir/
        images/      (fundus images)
        disc_masks/  (binary mask: disc=255, else 0)
        cup_masks/   (binary mask: cup=255, else 0)

    Filenames must match across the three folders.
    """

    def __init__(self, samples: List[Tuple[Path, Path, Path]], image_size: Tuple[int, int]):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, disc_path, cup_path = self.samples[idx]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        disc = cv2.imread(str(disc_path), cv2.IMREAD_GRAYSCALE)
        cup = cv2.imread(str(cup_path), cv2.IMREAD_GRAYSCALE)

        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        if disc is None:
            raise FileNotFoundError(f"Could not read disc mask: {disc_path}")
        if cup is None:
            raise FileNotFoundError(f"Could not read cup mask: {cup_path}")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_AREA)
        disc = cv2.resize(disc, self.image_size, interpolation=cv2.INTER_NEAREST)
        cup = cv2.resize(cup, self.image_size, interpolation=cv2.INTER_NEAREST)

        x = gray.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)  # (1, H, W)

        disc_bin = (disc > 127).astype(np.float32)
        cup_bin = (cup > 127).astype(np.float32)
        y = np.stack([disc_bin, cup_bin], axis=0)  # (2, H, W)

        return torch.from_numpy(x), torch.from_numpy(y)


def collect_samples(data_dir: Path) -> List[Tuple[Path, Path, Path]]:
    images_dir = data_dir / "images"
    disc_dir = data_dir / "disc_masks"
    cup_dir = data_dir / "cup_masks"

    if not images_dir.is_dir() or not disc_dir.is_dir() or not cup_dir.is_dir():
        raise FileNotFoundError(
            "Expected directories: images/, disc_masks/, cup_masks/ under --data-dir."
        )

    image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
    samples: List[Tuple[Path, Path, Path]] = []

    for image_path in image_files:
        disc_path = disc_dir / image_path.name
        cup_path = cup_dir / image_path.name
        if disc_path.is_file() and cup_path.is_file():
            samples.append((image_path, disc_path, cup_path))

    if not samples:
        raise RuntimeError("No matching image/disc/cup triplets found by filename.")

    return samples


def split_samples(
    samples: List[Tuple[Path, Path, Path]], val_split: float, seed: int
) -> Tuple[List[Tuple[Path, Path, Path]], List[Tuple[Path, Path, Path]]]:
    rng = random.Random(seed)
    samples_copy = samples[:]
    rng.shuffle(samples_copy)

    val_count = max(1, int(len(samples_copy) * val_split))
    val_samples = samples_copy[:val_count]
    train_samples = samples_copy[val_count:]
    if not train_samples:
        raise RuntimeError("Validation split too high; no training samples left.")
    return train_samples, val_samples


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_items += bs
    return total_loss / max(1, total_items)


def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)

    samples = collect_samples(cfg.data_dir)
    train_samples, val_samples = split_samples(samples, cfg.val_split, cfg.seed)

    train_ds = FundusSegmentationDataset(train_samples, cfg.image_size)
    val_ds = FundusSegmentationDataset(val_samples, cfg.image_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SimpleUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.output_path)
            print(f"Saved best weights to: {cfg.output_path}")

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train SimpleUNet for disc/cup segmentation and save .pth weights."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root directory.")
    parser.add_argument(
        "--output",
        type=str,
        default="glucoma/weights/best_model.pth",
        help="Path to save trained model weights.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split fraction (0,1)."
    )
    parser.add_argument("--device", type=str, default="cpu", help='Device: "cpu" or "cuda".')
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be between 0 and 1.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0.")

    return TrainConfig(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    train(parse_args())
