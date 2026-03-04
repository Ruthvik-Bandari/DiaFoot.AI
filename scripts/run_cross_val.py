"""DiaFoot.AI v2 — 5-Fold Cross Validation.

Trains U-Net++ segmentation on 5 folds for robust performance estimation.
Reports mean +/- std across folds.

Usage:
    python scripts/run_cross_val.py --fold 0 --device cuda --epochs 50
    (run with --fold 0,1,2,3,4 as SLURM array job)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.torch_dataset import DFUDataset
from src.evaluation.metrics import (
    aggregate_metrics,
    compute_segmentation_metrics,
)
from src.models.unetpp import build_unetpp
from src.training.losses import DiceCELoss
from src.training.schedulers import CosineAnnealingWithWarmup
from src.training.trainer import TrainConfig, Trainer


def create_fold_splits(
    train_csv: str | Path,
    val_csv: str | Path,
    fold: int,
    n_folds: int = 5,
    output_dir: str | Path = "data/splits/cv",
    filter_classes: list[str] | None = None,
) -> tuple[Path, Path]:
    """Create train/val split for a specific fold.

    Combines train+val, then splits into n_folds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    all_rows = []
    fieldnames = None
    for csv_path in [train_csv, val_csv]:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                if filter_classes and row.get("class", "") not in filter_classes:
                    continue
                all_rows.append(row)

    # Shuffle deterministically
    rng = np.random.RandomState(42)
    indices = list(range(len(all_rows)))
    rng.shuffle(indices)

    # Split into folds
    fold_size = len(indices) // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < n_folds - 1 else len(indices)

    val_indices = set(indices[val_start:val_end])
    train_indices = [i for i in indices if i not in val_indices]

    # Write fold CSVs
    fold_train = output_dir / f"train_fold{fold}.csv"
    fold_val = output_dir / f"val_fold{fold}.csv"

    for out_path, idx_list in [(fold_train, train_indices), (fold_val, list(val_indices))]:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or [])
            writer.writeheader()
            for i in idx_list:
                writer.writerow(all_rows[i])

    return fold_train, fold_val


def train_fold(fold: int, args: argparse.Namespace) -> dict:
    """Train and evaluate one fold."""
    logger = logging.getLogger(f"fold_{fold}")
    logger.info("Starting fold %d/%d", fold + 1, 5)

    # Create fold splits
    fold_train, fold_val = create_fold_splits(
        Path(args.splits_dir) / "train.csv",
        Path(args.splits_dir) / "val.csv",
        fold=fold,
        filter_classes=["dfu", "non_dfu"],
    )

    train_ds = DFUDataset(str(fold_train), transform=get_train_transforms())
    val_ds = DFUDataset(str(fold_val), transform=get_val_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    logger.info("Fold %d: %d train, %d val samples", fold, len(train_ds), len(val_ds))

    # Model
    model = build_unetpp(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        classes=1,
        decoder_attention_type="scse",
    )

    loss_fn = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.epochs,
    )
    torch.manual_seed(42 + fold)

    config = TrainConfig(
        epochs=args.epochs,
        precision="bf16-mixed",
        compile_model=False,
        gradient_clip=1.0,
        checkpoint_dir=f"checkpoints/cv_fold{fold}",
        monitor_metric="val/loss",
        monitor_mode="min",
        device=args.device,
        early_stopping_patience=15,
    )

    trainer = Trainer(model=model, config=config)
    trainer.fit(train_loader, val_loader, loss_fn, optimizer, scheduler)

    # Evaluate on fold validation set
    model.eval()
    fold_metrics = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(args.device)
            masks = batch["mask"].numpy()
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            for i in range(len(images)):
                m = compute_segmentation_metrics(preds[i], masks[i])
                fold_metrics.append(m)

    summary = aggregate_metrics(fold_metrics)
    dice = summary.get("dice", {}).get("mean", 0)
    iou = summary.get("iou", {}).get("mean", 0)
    logger.info("Fold %d results: Dice=%.4f, IoU=%.4f", fold, dice, iou)

    return {"fold": fold, "dice": dice, "iou": iou, "n_val": len(val_ds)}


def main() -> None:
    """Run cross-validation."""
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    result = train_fold(args.fold, args)

    # Save fold result
    output = Path(f"results/cv_fold{args.fold}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
