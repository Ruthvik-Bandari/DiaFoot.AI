"""DiaFoot.AI v2 — Data Composition Ablation.

The most important experiment: prove that adding healthy + non-DFU data helps.

Trains 3 segmentation models:
    (a) DFU-only: Train only on DFU images
    (b) DFU + non-DFU: Train on DFU + non-DFU (current best)
    (c) All: Train on all three classes (including healthy)

Usage:
    python scripts/run_ablation.py --variant dfu_only --device cuda --epochs 50
    python scripts/run_ablation.py --variant dfu_nondfu --device cuda --epochs 50
    python scripts/run_ablation.py --variant all --device cuda --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import build_dataloaders
from src.models.unetpp import build_unetpp
from src.training.losses import DiceCELoss
from src.training.schedulers import CosineAnnealingWithWarmup
from src.training.trainer import TrainConfig, Trainer

ABLATION_CONFIGS = {
    "dfu_only": {
        "classes": ["dfu"],
        "checkpoint_dir": "checkpoints/ablation_dfu_only",
        "description": "DFU images only (no negatives)",
    },
    "dfu_nondfu": {
        "classes": ["dfu", "non_dfu"],
        "checkpoint_dir": "checkpoints/ablation_dfu_nondfu",
        "description": "DFU + non-DFU wounds (current approach)",
    },
    "all": {
        "classes": None,  # No filter = all classes
        "checkpoint_dir": "checkpoints/ablation_all",
        "description": "All classes including healthy",
    },
}


def main() -> None:
    """Run data composition ablation."""
    parser = argparse.ArgumentParser(description="Data Composition Ablation")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(ABLATION_CONFIGS.keys()),
    )
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
    logger = logging.getLogger("ablation")

    config = ABLATION_CONFIGS[args.variant]
    logger.info("Ablation: %s — %s", args.variant, config["description"])

    model = build_unetpp(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        classes=1,
        decoder_attention_type="scse",
    )

    train_loader, val_loader = build_dataloaders(
        args.splits_dir,
        args.batch_size,
        args.num_workers,
        filter_classes=config["classes"],
    )
    logger.info(
        "Data: %d train, %d val batches",
        len(train_loader),
        len(val_loader),
    )

    loss_fn = DiceCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=5,
        max_epochs=args.epochs,
    )

    torch.manual_seed(42)

    trainer_config = TrainConfig(
        epochs=args.epochs,
        precision="bf16-mixed",
        compile_model=False,
        gradient_clip=1.0,
        checkpoint_dir=config["checkpoint_dir"],
        monitor_metric="val/loss",
        monitor_mode="min",
        device=args.device,
        early_stopping_patience=15,
    )

    trainer = Trainer(model=model, config=trainer_config)
    trainer.fit(train_loader, val_loader, loss_fn, optimizer, scheduler)
    logger.info("Ablation %s complete.", args.variant)


if __name__ == "__main__":
    main()
