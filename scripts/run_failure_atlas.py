"""DiaFoot.AI v2 — Generate failure atlas from segmentation predictions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_val_transforms
from src.data.torch_dataset import DFUDataset
from src.evaluation.failure_atlas import classify_segmentation_failure, summarize_failure_types
from src.evaluation.metrics import compute_segmentation_metrics
from src.models.unetpp import build_unetpp


def _to_u8_mask(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmentation failure atlas")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--output-dir", type=str, default="results/failure_atlas")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("failure_atlas")

    dev = args.device if torch.cuda.is_available() else "cpu"

    model = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(dev).eval()

    ds = DFUDataset(args.split_csv, transform=get_val_transforms(), return_metadata=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    rows = []
    idx_global = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(dev)
            masks = batch["mask"].numpy()
            metadata = batch["metadata"]
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs > 0.5).astype(np.uint8)

            for i in range(len(images)):
                gt = masks[i]
                pred = preds[i]
                m = compute_segmentation_metrics(pred, gt)
                ftype = classify_segmentation_failure(pred, gt, m["dice"])
                filename = metadata["filename"][i] if "filename" in metadata else f"sample_{idx_global}.png"
                rows.append(
                    {
                        "index": idx_global,
                        "filename": filename,
                        "dice": float(m["dice"]),
                        "iou": float(m["iou"]),
                        "hd95": float(m["hd95"]),
                        "failure_type": ftype,
                        "pred": pred,
                        "gt": gt,
                    }
                )
                idx_global += 1

    # Worst examples by Dice
    rows_sorted = sorted(rows, key=lambda r: r["dice"])
    selected = rows_sorted[: max(1, args.top_k)]

    out_dir = Path(args.output_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    manifest = []
    for r in selected:
        stem = f"{r['index']:05d}_{Path(str(r['filename'])).stem}"
        pred_path = out_dir / "images" / f"{stem}_pred.png"
        gt_path = out_dir / "images" / f"{stem}_gt.png"
        cv2.imwrite(str(pred_path), _to_u8_mask(r["pred"]))
        cv2.imwrite(str(gt_path), _to_u8_mask(r["gt"]))

        manifest.append(
            {
                "index": r["index"],
                "filename": r["filename"],
                "dice": r["dice"],
                "iou": r["iou"],
                "hd95": r["hd95"],
                "failure_type": r["failure_type"],
                "pred_mask": str(pred_path),
                "gt_mask": str(gt_path),
            }
        )

    summary = summarize_failure_types([r["failure_type"] for r in rows])
    report = {
        "num_samples": len(rows),
        "selected_top_k": len(selected),
        "failure_summary": summary,
        "examples": manifest,
    }

    report_path = out_dir / "failure_atlas.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Failure atlas saved to %s", report_path)


if __name__ == "__main__":
    main()
