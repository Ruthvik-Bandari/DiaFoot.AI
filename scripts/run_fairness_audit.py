"""DiaFoot.AI v2 — Run Fairness Audit on Trained Models.

Connects ITA skin tone scores to actual model predictions.

Usage:
    python scripts/run_fairness_audit.py \
        --seg-checkpoint checkpoints/dinov2_segmenter/best.pt \
        --cls-checkpoint checkpoints/dinov2_classifier/best.pt
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
from src.evaluation.fairness import (
    print_fairness_report,
    run_fairness_audit,
)
from src.evaluation.metrics import compute_segmentation_metrics
from src.models.classifier import TriageClassifier
from src.models.unetpp import build_unetpp


def main() -> None:
    """Run fairness audit on trained models."""
    parser = argparse.ArgumentParser(description="Fairness Audit")
    parser.add_argument("--cls-checkpoint", type=str, default=None)
    parser.add_argument("--seg-checkpoint", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--ita-csv", type=str, default="data/metadata/ita_scores.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("fairness")

    dev = args.device if torch.cuda.is_available() else "cpu"
    test_ds = DFUDataset(
        Path(args.splits_dir) / "test.csv",
        transform=get_val_transforms(),
        return_metadata=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=2,
    )

    cls_results = None
    seg_results = None

    # Classification audit
    if args.cls_checkpoint and Path(args.cls_checkpoint).exists():
        logger.info("Loading classifier: %s", args.cls_checkpoint)
        model = TriageClassifier(
            backbone="tf_efficientnetv2_m",
            num_classes=3,
            pretrained=False,
        )
        ckpt = torch.load(args.cls_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dev).eval()

        filenames, y_true, y_pred = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(dev)
                labels = batch["label"]
                preds = model(imgs).argmax(dim=1).cpu()
                for i in range(len(labels)):
                    filenames.append(batch["metadata"]["filename"][i])
                    y_true.append(labels[i].item())
                    y_pred.append(preds[i].item())

        cls_results = {
            "filenames": filenames,
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
        }
        logger.info("Classification: %d predictions collected", len(filenames))

    # Segmentation audit
    if args.seg_checkpoint and Path(args.seg_checkpoint).exists():
        logger.info("Loading segmenter: %s", args.seg_checkpoint)
        model = build_unetpp(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            classes=1,
        )
        ckpt = torch.load(args.seg_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(dev).eval()

        filenames, metrics_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(dev)
                masks = batch["mask"].numpy()
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
                for i in range(len(imgs)):
                    filenames.append(batch["metadata"]["filename"][i])
                    m = compute_segmentation_metrics(preds[i], masks[i])
                    metrics_list.append(m)

        seg_results = {
            "filenames": filenames,
            "metrics_per_image": metrics_list,
        }
        logger.info("Segmentation: %d predictions collected", len(filenames))

    # Run fairness audit
    report = run_fairness_audit(
        classification_results=cls_results,
        segmentation_results=seg_results,
        ita_csv=args.ita_csv,
    )
    print_fairness_report(report)

    # Save report
    output = Path("results/fairness_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=convert)
    logger.info("Report saved to %s", output)


if __name__ == "__main__":
    main()
