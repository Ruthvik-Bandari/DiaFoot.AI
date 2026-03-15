#!/usr/bin/env python3
"""Evaluate baseline U-Net++ checkpoint on a chosen split and class subset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.augmentation import get_val_transforms
from src.data.torch_dataset import CLASS_TO_IDX, DFUDataset
from src.evaluation.metrics import aggregate_metrics, compute_segmentation_metrics
from src.models.unetpp import build_unetpp

logger = logging.getLogger("evaluate_baseline_split")


def parse_include_classes(raw: str) -> set[int] | None:
    if raw.strip().lower() == "all":
        return None
    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    labels: set[int] = set()
    for name in names:
        if name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class '{name}'. Valid: {list(CLASS_TO_IDX.keys())} or 'all'")
        labels.add(CLASS_TO_IDX[name])
    return labels


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate baseline segmentation checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--include-classes", default="dfu,non_dfu", help="Comma-separated class names or 'all'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", default="results/segmentation_metrics_baseline_matched.json")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    include_labels = parse_include_classes(args.include_classes)

    model = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    ds = DFUDataset(
        split_csv=Path(args.splits_dir) / f"{args.split}.csv",
        transform=get_val_transforms(),
        return_metadata=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = []
    selected = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            labels = batch["label"].numpy()

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

            for i in range(len(images)):
                if include_labels is not None and int(labels[i]) not in include_labels:
                    continue
                m = compute_segmentation_metrics(preds[i], masks[i])
                metrics.append(m)
                selected += 1

    if not metrics:
        raise RuntimeError("No samples selected for evaluation. Check --split and --include-classes.")

    summary = aggregate_metrics(metrics)
    output = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "include_classes": args.include_classes,
        "num_samples": selected,
        "metrics": summary,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))

    logger.info("Saved baseline matched eval to %s", out_path)
    print("\n=== Baseline Matched Evaluation ===")
    print(f"samples: {selected}")
    for k in ("dice", "iou", "hd95", "nsd_2mm", "nsd_5mm"):
        if k in summary:
            print(f"{k:>8}: {summary[k]['mean']:.4f}")


if __name__ == "__main__":
    main()
