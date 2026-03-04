"""DiaFoot.AI v2 — Run TTA Evaluation.

Compares model performance with and without Test-Time Augmentation.

Usage:
    python scripts/run_tta_eval.py \
        --checkpoint checkpoints/segmentation_v2/best_epoch018_0.4109.pt \
        --device cuda
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
from src.inference.tta import tta_predict_segmentation
from src.models.unetpp import build_unetpp


def main() -> None:
    """Compare base vs TTA predictions."""
    parser = argparse.ArgumentParser(description="TTA Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-augmentations", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("tta_eval")

    dev = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_unetpp(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        classes=1,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dev).eval()

    # Load test data
    test_ds = DFUDataset(
        Path(args.splits_dir) / "test.csv",
        transform=get_val_transforms(),
    )

    base_metrics = []
    tta_metrics = []
    n_images = min(args.max_images, len(test_ds))

    logger.info(
        "Evaluating %d images: base vs TTA (%d augmentations)",
        n_images,
        args.num_augmentations,
    )

    for idx in range(n_images):
        sample = test_ds[idx]
        image = sample["image"].unsqueeze(0).to(dev)
        gt_mask = sample["mask"].numpy()

        # Base prediction (no TTA)
        with torch.no_grad():
            logits = model(image)
            base_pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        base_m = compute_segmentation_metrics(base_pred, gt_mask)
        base_metrics.append(base_m)

        # TTA prediction
        tta_pred, _uncertainty = tta_predict_segmentation(
            model,
            image,
            device=dev,
            num_augmentations=args.num_augmentations,
        )
        tta_mask = (tta_pred > 0.5).astype(np.uint8)
        tta_m = compute_segmentation_metrics(tta_mask, gt_mask)
        tta_metrics.append(tta_m)

        if (idx + 1) % 20 == 0:
            logger.info("  Processed %d/%d images", idx + 1, n_images)

    # Aggregate
    base_summary = aggregate_metrics(base_metrics)
    tta_summary = aggregate_metrics(tta_metrics)

    print(f"\n{'=' * 70}")
    print("TTA Evaluation Results")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} {'Base':>10} {'TTA':>10} {'Diff':>10}")
    print("-" * 70)

    for key in ["dice", "iou", "hd95", "nsd_2mm", "nsd_5mm"]:
        base_val = base_summary.get(key, {}).get("mean", 0)
        tta_val = tta_summary.get(key, {}).get("mean", 0)
        diff = tta_val - base_val
        sign = "+" if diff >= 0 else ""
        print(f"  {key:<18} {base_val:>9.4f} {tta_val:>9.4f} {sign}{diff:>9.4f}")

    print(f"{'=' * 70}\n")

    # Save results
    output = Path("results/tta_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "base": {
                    k: v.get("mean", 0) for k, v in base_summary.items() if isinstance(v, dict)
                },
                "tta": {k: v.get("mean", 0) for k, v in tta_summary.items() if isinstance(v, dict)},
                "num_augmentations": args.num_augmentations,
                "num_images": n_images,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
