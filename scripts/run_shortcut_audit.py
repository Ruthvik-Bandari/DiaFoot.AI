"""DiaFoot.AI v2 — Classifier shortcut-learning audit.

Perturbs image background/borders and quantifies prediction drift.
A large drift suggests shortcut reliance on non-clinical cues.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import TYPE_CHECKING

from src.data.torch_dataset import CLASS_TO_IDX
from src.evaluation.shortcut_audit import (
    blur_background,
    keep_center_only,
    perturb_border_noise,
    summarize_shortcut_shift,
)
from src.inference.pipeline import InferencePipeline
from src.models.classifier import TriageClassifier

if TYPE_CHECKING:
    from collections.abc import Callable


def _read_split(split_csv: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(split_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _build_predictor(
    checkpoint: str | Path, device: str
) -> Callable[[np.ndarray], tuple[int, float]]:
    model = TriageClassifier(backbone="tf_efficientnetv2_m", num_classes=3, pretrained=False)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    pipe = InferencePipeline(classifier=model, segmenter=None, device=device)

    @torch.no_grad()
    def predict(image: np.ndarray) -> tuple[int, float]:
        t = pipe.preprocess(image).to(pipe.device)
        logits = pipe.classifier(t)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return int(probs.argmax()), float(probs.max())

    return predict


def _evaluate_perturbation(
    rows: list[dict[str, str]],
    predictor: Callable[[np.ndarray], tuple[int, float]],
    perturb_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float | int]:
    labels: list[int] = []
    base_pred: list[int] = []
    base_conf: list[float] = []
    pert_pred: list[int] = []
    pert_conf: list[float] = []

    for row in rows:
        image_path = row.get("image_path") or row.get("image")
        class_name = row.get("class", "healthy")
        if not image_path:
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue

        y = CLASS_TO_IDX.get(class_name, 0)
        pred0, conf0 = predictor(image)
        image_pert = perturb_fn(image)
        pred1, conf1 = predictor(image_pert)

        labels.append(y)
        base_pred.append(pred0)
        base_conf.append(conf0)
        pert_pred.append(pred1)
        pert_conf.append(conf1)

    if not labels:
        return {
            "n": 0,
            "baseline_accuracy": 0.0,
            "perturbed_accuracy": 0.0,
            "accuracy_drop": 0.0,
            "prediction_consistency": 0.0,
            "confidence_drop_mean": 0.0,
            "confidence_drop_median": 0.0,
            "confidence_drop_p90": 0.0,
        }

    return summarize_shortcut_shift(
        labels=np.array(labels),
        baseline_pred=np.array(base_pred),
        baseline_conf=np.array(base_conf),
        perturbed_pred=np.array(pert_pred),
        perturbed_conf=np.array(pert_conf),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifier shortcut-learning audit")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/classifier/best_epoch004_1.0000.pt",
    )
    parser.add_argument("--split-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/shortcut_audit.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("shortcut_audit")

    dev = args.device if torch.cuda.is_available() else "cpu"
    rows = _read_split(args.split_csv, limit=args.max_samples)
    predictor = _build_predictor(args.checkpoint, dev)

    logger.info("Running shortcut audit on %d samples", len(rows))
    report = {
        "split_csv": args.split_csv,
        "checkpoint": args.checkpoint,
        "num_samples_requested": args.max_samples,
        "noise_border": _evaluate_perturbation(rows, predictor, perturb_border_noise),
        "center_only": _evaluate_perturbation(rows, predictor, keep_center_only),
        "blur_background": _evaluate_perturbation(rows, predictor, blur_background),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Shortcut audit saved to %s", out_path)


if __name__ == "__main__":
    main()
