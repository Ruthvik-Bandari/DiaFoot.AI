"""DiaFoot.AI v2 — Improved Training Entry Point.

Key fix: Segmentation trains ONLY on images with actual wounds (DFU + non-DFU).
Healthy images with empty masks are excluded from segmentation training.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.torch_dataset import DFUDataset
from src.models.classifier import TriageClassifier
from src.models.unetpp import build_unetpp
from src.training.classification_losses import FocalLoss
from src.training.losses import DiceCELoss
from src.training.schedulers import CosineAnnealingWithWarmup
from src.training.trainer import TrainConfig, Trainer


def filter_split_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    exclude_classes: list[str] | None = None,
    include_classes: list[str] | None = None,
) -> int:
    """Filter a split CSV to include/exclude specific classes.

    Args:
        input_csv: Original split CSV.
        output_csv: Filtered output CSV.
        exclude_classes: Classes to exclude.
        include_classes: Classes to include (overrides exclude).

    Returns:
        Number of rows in filtered CSV.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            cls = row.get("class", "")
            if include_classes and cls not in include_classes:
                continue
            if exclude_classes and cls in exclude_classes:
                continue
            rows.append(row)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def build_dataloaders(
    splits_dir: str,
    batch_size: int,
    num_workers: int,
    filter_classes: list[str] | None = None,
) -> tuple:
    """Build data loaders, optionally filtering by class."""
    splits_path = Path(splits_dir)

    if filter_classes:
        # Create filtered CSVs
        filtered_dir = splits_path / "filtered"
        filtered_dir.mkdir(exist_ok=True)
        train_csv = filtered_dir / "train.csv"
        val_csv = filtered_dir / "val.csv"
        n_train = filter_split_csv(
            splits_path / "train.csv",
            train_csv,
            include_classes=filter_classes,
        )
        n_val = filter_split_csv(
            splits_path / "val.csv",
            val_csv,
            include_classes=filter_classes,
        )
        logging.getLogger("train").info(
            "Filtered data: %d train, %d val (classes: %s)",
            n_train,
            n_val,
            filter_classes,
        )
    else:
        train_csv = splits_path / "train.csv"
        val_csv = splits_path / "val.csv"

    train_ds = DFUDataset(str(train_csv), transform=get_train_transforms())
    val_ds = DFUDataset(str(val_csv), transform=get_val_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def train_classifier(args: argparse.Namespace) -> None:
    """Train triage classifier on ALL classes."""
    logger = logging.getLogger("train_classifier")

    model = TriageClassifier(
        backbone="tf_efficientnetv2_m",
        num_classes=3,
        dropout=0.3,
        pretrained=True,
    )
    logger.info("Model: TriageClassifier (tf_efficientnetv2_m)")

    train_loader, val_loader = build_dataloaders(
        args.splits_dir,
        args.batch_size,
        args.num_workers,
    )
    logger.info("Data: %d train, %d val batches (all classes)", len(train_loader), len(val_loader))

    loss_fn = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=5, max_epochs=args.epochs)

    config = TrainConfig(
        epochs=args.epochs,
        precision="bf16-mixed",
        compile_model=False,
        gradient_clip=1.0,
        checkpoint_dir="checkpoints/classifier",
        monitor_metric="val/accuracy",
        monitor_mode="max",
        device=args.device,
    )
    trainer = Trainer(model=model, config=config)
    trainer.fit(train_loader, val_loader, loss_fn, optimizer, scheduler)


def train_segmentation(args: argparse.Namespace) -> None:
    """Train segmentation on WOUND IMAGES ONLY (DFU + non-DFU).

    KEY FIX: Excludes healthy feet (empty masks) from training.
    This prevents the model from learning 'predict nothing' on 47% of data.
    """
    logger = logging.getLogger("train_segmentation")

    model = build_unetpp(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        classes=1,
        decoder_attention_type="scse",
    )
    logger.info("Model: U-Net++ (efficientnet-b4)")

    # CRITICAL: Only train on images with actual wounds
    train_loader, val_loader = build_dataloaders(
        args.splits_dir,
        args.batch_size,
        args.num_workers,
        filter_classes=["dfu", "non_dfu"],  # Exclude healthy
    )
    logger.info(
        "Data: %d train, %d val batches (DFU + non-DFU only, healthy excluded)",
        len(train_loader),
        len(val_loader),
    )

    loss_fn = DiceCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=5, max_epochs=args.epochs)

    config = TrainConfig(
        epochs=args.epochs,
        precision="bf16-mixed",
        compile_model=False,
        gradient_clip=1.0,
        checkpoint_dir="checkpoints/segmentation_v2",
        monitor_metric="val/loss",
        monitor_mode="min",
        device=args.device,
    )
    trainer = Trainer(model=model, config=config)
    trainer.fit(train_loader, val_loader, loss_fn, optimizer, scheduler)


def main() -> None:
    """Run training."""
    parser = argparse.ArgumentParser(description="DiaFoot.AI v2 Training (Improved)")
    parser.add_argument("--task", type=str, required=True, choices=["classify", "segment"])
    parser.add_argument("--config", type=str, default="configs/training/baseline.yaml")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(42)

    if args.task == "classify":
        train_classifier(args)
    elif args.task == "segment":
        train_segmentation(args)


if __name__ == "__main__":
    main()
