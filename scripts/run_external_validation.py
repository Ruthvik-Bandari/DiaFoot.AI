"""DiaFoot.AI v2 — External Validation Benchmark.

Compares internal test split vs external holdout split and reports performance drop.

Usage:
    python scripts/run_external_validation.py \
        --internal-split data/splits/test.csv \
        --external-split data/splits/external.csv \
        --cls-checkpoint checkpoints/classifier/best_epoch004_1.0000.pt \
        --seg-checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt
"""

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
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.external_validation import bootstrap_ci, compute_drop_report
from src.evaluation.metrics import compute_segmentation_metrics
from src.models.classifier import TriageClassifier
from src.models.unetpp import build_unetpp


def _evaluate_classifier(
    split_csv: str | Path,
    checkpoint: str | Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    model = TriageClassifier(backbone="tf_efficientnetv2_m", num_classes=3, pretrained=False)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    ds = DFUDataset(split_csv=split_csv, transform=get_val_transforms(), return_metadata=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(list(probs))

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_prob_np = np.array(y_prob)

    metrics = compute_classification_metrics(y_true_np, y_pred_np, y_prob_np)
    acc_values = (y_true_np == y_pred_np).astype(float)
    metrics["accuracy_ci95"] = bootstrap_ci(acc_values)
    return metrics


def _evaluate_segmentation(
    split_csv: str | Path,
    checkpoint: str | Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    model = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    ds = DFUDataset(split_csv=split_csv, transform=get_val_transforms(), return_metadata=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_metrics: list[dict[str, float]] = []
    dfu_metrics: list[dict[str, float]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

            for i in range(len(images)):
                m = compute_segmentation_metrics(preds[i], masks[i])
                all_metrics.append(m)
                if labels[i] == 2:
                    dfu_metrics.append(m)

    def _summarize(metrics_list: list[dict[str, float]]) -> dict[str, Any]:
        if not metrics_list:
            return {"count": 0}
        dice = np.array([m["dice"] for m in metrics_list], dtype=float)
        iou = np.array([m["iou"] for m in metrics_list], dtype=float)
        hd95 = np.array([m["hd95"] for m in metrics_list], dtype=float)
        out = {
            "count": len(metrics_list),
            "dice": float(dice.mean()),
            "iou": float(iou.mean()),
            "hd95": float(hd95.mean()),
            "dice_ci95": bootstrap_ci(dice),
            "iou_ci95": bootstrap_ci(iou),
        }
        return out

    return {
        "overall": _summarize(all_metrics),
        "dfu_only": _summarize(dfu_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation benchmark")
    parser.add_argument("--internal-split", type=str, default="data/splits/test.csv")
    parser.add_argument("--external-split", type=str, required=True)
    parser.add_argument("--cls-checkpoint", type=str, default="")
    parser.add_argument("--seg-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/external_validation_report.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("external_validation")

    dev = args.device if torch.cuda.is_available() else "cpu"
    report: dict[str, Any] = {
        "internal_split": args.internal_split,
        "external_split": args.external_split,
    }

    if args.cls_checkpoint:
        cls_ckpt = Path(args.cls_checkpoint)
        if cls_ckpt.exists():
            logger.info("Evaluating classifier on internal split")
            cls_internal = _evaluate_classifier(
                split_csv=args.internal_split,
                checkpoint=cls_ckpt,
                device=dev,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            logger.info("Evaluating classifier on external split")
            cls_external = _evaluate_classifier(
                split_csv=args.external_split,
                checkpoint=cls_ckpt,
                device=dev,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            cls_drop = compute_drop_report(
                internal={
                    "accuracy": cls_internal.get("accuracy", 0.0),
                    "f1_macro": cls_internal.get("f1_macro", 0.0),
                    "dfu_sensitivity": cls_internal.get("dfu_sensitivity", 0.0),
                    "healthy_specificity": cls_internal.get("healthy_specificity", 0.0),
                },
                external={
                    "accuracy": cls_external.get("accuracy", 0.0),
                    "f1_macro": cls_external.get("f1_macro", 0.0),
                    "dfu_sensitivity": cls_external.get("dfu_sensitivity", 0.0),
                    "healthy_specificity": cls_external.get("healthy_specificity", 0.0),
                },
                keys=["accuracy", "f1_macro", "dfu_sensitivity", "healthy_specificity"],
            )
            report["classification"] = {
                "internal": cls_internal,
                "external": cls_external,
                "drop": cls_drop,
            }
        else:
            logger.warning("Classifier checkpoint not found: %s", cls_ckpt)

    if args.seg_checkpoint:
        seg_ckpt = Path(args.seg_checkpoint)
        if seg_ckpt.exists():
            logger.info("Evaluating segmentation on internal split")
            seg_internal = _evaluate_segmentation(
                split_csv=args.internal_split,
                checkpoint=seg_ckpt,
                device=dev,
                batch_size=max(1, args.batch_size // 2),
                num_workers=args.num_workers,
            )
            logger.info("Evaluating segmentation on external split")
            seg_external = _evaluate_segmentation(
                split_csv=args.external_split,
                checkpoint=seg_ckpt,
                device=dev,
                batch_size=max(1, args.batch_size // 2),
                num_workers=args.num_workers,
            )
            seg_drop = compute_drop_report(
                internal={
                    "dice": seg_internal.get("dfu_only", {}).get("dice", 0.0),
                    "iou": seg_internal.get("dfu_only", {}).get("iou", 0.0),
                    "hd95": seg_internal.get("dfu_only", {}).get("hd95", 0.0),
                },
                external={
                    "dice": seg_external.get("dfu_only", {}).get("dice", 0.0),
                    "iou": seg_external.get("dfu_only", {}).get("iou", 0.0),
                    "hd95": seg_external.get("dfu_only", {}).get("hd95", 0.0),
                },
                keys=["dice", "iou", "hd95"],
            )
            report["segmentation"] = {
                "internal": seg_internal,
                "external": seg_external,
                "drop": seg_drop,
            }
        else:
            logger.warning("Segmentation checkpoint not found: %s", seg_ckpt)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("External validation report saved to %s", out_path)


if __name__ == "__main__":
    main()
