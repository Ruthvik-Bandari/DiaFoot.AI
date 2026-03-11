"""DiaFoot.AI v2 — Subgroup audit (ITA/source/wound-size) with CIs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_val_transforms
from src.data.torch_dataset import DFUDataset
from src.evaluation.metrics import compute_segmentation_metrics
from src.evaluation.subgroup_analysis import (
    classification_subgroup_report,
    segmentation_subgroup_report,
)
from src.models.classifier import TriageClassifier
from src.models.unetpp import build_unetpp


def _wound_size_group(mask: np.ndarray) -> str:
    area = float(mask.astype(bool).sum())
    total = float(mask.shape[0] * mask.shape[1])
    ratio = area / total if total > 0 else 0.0
    if ratio < 0.01:
        return "small"
    if ratio < 0.05:
        return "medium"
    return "large"


def main() -> None:
    parser = argparse.ArgumentParser(description="Subgroup analysis with confidence intervals")
    parser.add_argument("--split-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--cls-checkpoint", type=str, default="")
    parser.add_argument("--seg-checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=str, default="results/subgroup_audit.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("subgroup_audit")

    dev = args.device if torch.cuda.is_available() else "cpu"

    ds = DFUDataset(split_csv=args.split_csv, transform=get_val_transforms(), return_metadata=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Collect metadata-aligned vectors.
    labels: list[int] = []
    ita_groups: list[str] = []
    source_groups: list[str] = []
    wound_size_groups: list[str] = []

    for row in ds.samples:
        labels.append({"healthy": 0, "non_dfu": 1, "dfu": 2}.get(row.get("class", "healthy"), 0))
        ita_groups.append(row.get("ita_category") or row.get("ita_group") or "Unknown")
        source_groups.append(row.get("source_id") or row.get("class") or "unknown")

    # Need mask-derived wound size from dataset order
    for i in range(len(ds)):
        sample = ds[i]
        mask = (
            sample["mask"].numpy() if hasattr(sample["mask"], "numpy") else np.array(sample["mask"])
        )
        wound_size_groups.append(_wound_size_group(mask))

    report: dict[str, Any] = {
        "split_csv": args.split_csv,
        "groups": {
            "ita": sorted(set(ita_groups)),
            "source": sorted(set(source_groups)),
            "wound_size": sorted(set(wound_size_groups)),
        },
    }

    # Classification subgroup audit
    if args.cls_checkpoint and Path(args.cls_checkpoint).exists():
        logger.info("Running classification subgroup audit")
        model = TriageClassifier(backbone="tf_efficientnetv2_m", num_classes=3, pretrained=False)
        ckpt = torch.load(args.cls_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dev).eval()

        preds: list[int] = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(dev)
                p = model(images).argmax(dim=1).cpu().numpy()
                preds.extend(p.tolist())

        y_true = np.array(labels)
        y_pred = np.array(preds)
        report["classification"] = {
            "ita": classification_subgroup_report(y_true, y_pred, ita_groups),
            "source": classification_subgroup_report(y_true, y_pred, source_groups),
            "wound_size": classification_subgroup_report(y_true, y_pred, wound_size_groups),
        }

    # Segmentation subgroup audit
    if args.seg_checkpoint and Path(args.seg_checkpoint).exists():
        logger.info("Running segmentation subgroup audit")
        model = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
        ckpt = torch.load(args.seg_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dev).eval()

        dice_vals: list[float] = []
        iou_vals: list[float] = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(dev)
                masks = batch["mask"].numpy()
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
                for i in range(len(images)):
                    m = compute_segmentation_metrics(preds[i], masks[i])
                    dice_vals.append(m["dice"])
                    iou_vals.append(m["iou"])

        dice_np = np.array(dice_vals)
        iou_np = np.array(iou_vals)
        report["segmentation"] = {
            "ita": segmentation_subgroup_report(dice_np, iou_np, ita_groups),
            "source": segmentation_subgroup_report(dice_np, iou_np, source_groups),
            "wound_size": segmentation_subgroup_report(dice_np, iou_np, wound_size_groups),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Subgroup audit saved to %s", out)


if __name__ == "__main__":
    main()
