"""DiaFoot.AI v2 — Evaluation Entry Point.

Phase 4: Evaluate trained models on test set.

Usage:
    # Evaluate classifier
    python scripts/evaluate.py --task classify \

    # Evaluate segmentation
    python scripts/evaluate.py --task segment \
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
from src.evaluation.calibration import compute_calibration_report, print_calibration_report
from src.evaluation.classification_metrics import (
    compute_classification_metrics,
    print_classification_report,
)
from src.evaluation.metrics import (
    aggregate_metrics,
    compute_segmentation_metrics,
    print_segmentation_report,
)
from src.models.classifier import TriageClassifier
from src.models.unetpp import build_unetpp


def evaluate_classifier(checkpoint_path: str, splits_dir: str, device: str) -> None:
    """Evaluate triage classifier on test set."""
    logger = logging.getLogger("eval_classifier")

    model = TriageClassifier(backbone="tf_efficientnetv2_m", num_classes=3, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_ds = DFUDataset(
        split_csv=Path(splits_dir) / "test.csv",
        transform=get_val_transforms(),
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    all_labels = []
    all_preds = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_logits = np.array(all_logits)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print_classification_report(metrics)

    calibration_report = compute_calibration_report(
        classification_logits=y_logits,
        classification_labels=y_true,
    )
    print_calibration_report(calibration_report)

    # Save results
    output_path = Path("results/classification_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics = {k: v for k, v in metrics.items() if k != "report"}
    if "classification" in calibration_report:
        save_metrics["calibration"] = calibration_report["classification"]
    with open(output_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    logger.info("Results saved to %s", output_path)

    calibration_path = Path("results/classification_calibration.json")
    with open(calibration_path, "w") as f:
        json.dump(calibration_report, f, indent=2)
    logger.info("Calibration report saved to %s", calibration_path)


def evaluate_segmentation(checkpoint_path: str, splits_dir: str, device: str) -> None:
    """Evaluate segmentation model on test set."""
    logger = logging.getLogger("eval_segmentation")

    model = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_ds = DFUDataset(
        split_csv=Path(splits_dir) / "test.csv",
        transform=get_val_transforms(),
        return_metadata=True,
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    all_metrics = []
    dfu_metrics = []
    non_dfu_metrics = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

            for i in range(len(images)):
                pred_mask = preds[i]
                gt_mask = masks[i]
                m = compute_segmentation_metrics(pred_mask, gt_mask)
                all_metrics.append(m)

                if labels[i] == 2:
                    dfu_metrics.append(m)
                elif labels[i] == 1:
                    non_dfu_metrics.append(m)

    # Overall results
    summary = aggregate_metrics(all_metrics)
    print_segmentation_report(summary)

    # Per-class results
    if dfu_metrics:
        print("DFU images only:")
        dfu_summary = aggregate_metrics(dfu_metrics)
        print_segmentation_report(dfu_summary)

    if non_dfu_metrics:
        print("Non-DFU images only:")
        non_dfu_summary = aggregate_metrics(non_dfu_metrics)
        print_segmentation_report(non_dfu_summary)

    # Save results
    output_path = Path("results/segmentation_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)


def main() -> None:
    """Run evaluation."""
    parser = argparse.ArgumentParser(description="DiaFoot.AI v2 Evaluation")
    parser.add_argument("--task", type=str, required=True, choices=["classify", "segment"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.task == "classify":
        evaluate_classifier(args.checkpoint, args.splits_dir, dev)
    elif args.task == "segment":
        evaluate_segmentation(args.checkpoint, args.splits_dir, dev)


if __name__ == "__main__":
    main()
