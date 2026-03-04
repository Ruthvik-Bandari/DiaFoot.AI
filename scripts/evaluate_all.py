"""DiaFoot.AI v2 — Evaluate All Models.

Evaluates U-Net++, FUSegNet, and ablation checkpoints.

Usage:
    python scripts/evaluate_all.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_val_transforms
from src.data.torch_dataset import DFUDataset
from src.evaluation.metrics import (
    aggregate_metrics,
    compute_segmentation_metrics,
)


def evaluate_checkpoint(
    model: torch.nn.Module,
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    device: str,
) -> dict:
    """Evaluate a single model checkpoint."""
    model = model.to(device).eval()

    all_metrics = []
    dfu_metrics = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()

            logits = model(images)
            if isinstance(logits, dict):
                logits = logits.get("seg_logits", logits)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

            for i in range(len(images)):
                m = compute_segmentation_metrics(preds[i], masks[i])
                all_metrics.append(m)
                if labels[i] == 2:
                    dfu_metrics.append(m)

    overall = aggregate_metrics(all_metrics)
    dfu_only = aggregate_metrics(dfu_metrics)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Overall: Dice={overall['dice']['mean']:.4f}  IoU={overall['iou']['mean']:.4f}")
    if dfu_only:
        print(
            f"  DFU:     Dice={dfu_only['dice']['mean']:.4f}  "
            f"IoU={dfu_only['iou']['mean']:.4f}  "
            f"NSD@5mm={dfu_only.get('nsd_5mm', {}).get('mean', 0):.4f}"
        )
    print(f"{'=' * 60}")

    return {"overall": overall, "dfu_only": dfu_only}


def main() -> None:
    """Evaluate all available models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits-dir", default="data/splits")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dev = args.device if torch.cuda.is_available() else "cpu"

    test_ds = DFUDataset(
        Path(args.splits_dir) / "test.csv",
        transform=get_val_transforms(),
        return_metadata=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    results = {}

    # Define all checkpoints to evaluate
    checkpoints = {
        "U-Net++ (DFU-only)": {
            "path": "checkpoints/ablation_dfu_only/best_epoch045_0.1057.pt",
            "model_type": "unetpp",
        },
        "U-Net++ (DFU+nonDFU)": {
            "path": "checkpoints/ablation_dfu_nondfu/best_epoch018_0.4187.pt",
            "model_type": "unetpp",
        },
        "U-Net++ (All classes)": {
            "path": "checkpoints/ablation_all/best_epoch029_0.6723.pt",
            "model_type": "unetpp",
        },
        "U-Net++ v2 (DFU+nonDFU, fixed)": {
            "path": "checkpoints/segmentation_v2/best_epoch018_0.4109.pt",
            "model_type": "unetpp",
        },
        "FUSegNet (DFU+nonDFU)": {
            "path": "checkpoints/fusegnet/best_epoch021_0.4171.pt",
            "model_type": "fusegnet",
        },
    }

    for name, config in checkpoints.items():
        ckpt_path = Path(config["path"])
        if not ckpt_path.exists():
            print(f"\n  SKIP: {name} — checkpoint not found: {ckpt_path}")
            continue

        # Build model
        if config["model_type"] == "unetpp":
            from src.models.unetpp import build_unetpp

            model = build_unetpp(
                encoder_name="efficientnet-b4",
                encoder_weights=None,
                classes=1,
            )
        elif config["model_type"] == "fusegnet":
            from src.models.fusegnet import FUSegNet

            model = FUSegNet(
                encoder_name="efficientnet-b7",
                encoder_weights=None,
                classes=1,
            )

        # Load checkpoint
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        results[name] = evaluate_checkpoint(model, name, test_loader, dev)

    # Print summary table
    print(f"\n\n{'=' * 80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Model':<35} {'DFU Dice':>10} {'DFU IoU':>10} {'DFU NSD@5':>10}")
    print("-" * 80)
    for name, res in results.items():
        dfu = res.get("dfu_only", {})
        dice = dfu.get("dice", {}).get("mean", 0)
        iou = dfu.get("iou", {}).get("mean", 0)
        nsd = dfu.get("nsd_5mm", {}).get("mean", 0)
        print(f"  {name:<33} {dice:>9.4f} {iou:>9.4f} {nsd:>9.4f}")
    print(f"{'=' * 80}")

    # Save results
    output = Path("results/all_models_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, res in results.items():
        save_data[name] = {
            "dfu_dice": res.get("dfu_only", {}).get("dice", {}).get("mean", 0),
            "dfu_iou": res.get("dfu_only", {}).get("iou", {}).get("mean", 0),
            "dfu_nsd5mm": res.get("dfu_only", {}).get("nsd_5mm", {}).get("mean", 0),
        }
    with open(output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
