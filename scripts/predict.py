"""DiaFoot.AI v2 — Inference on New Images.

Run the trained pipeline on any foot image.

Usage:
    python scripts/predict.py --image path/to/foot_image.jpg
    python scripts/predict.py --image path/to/image.jpg --save-mask output_mask.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_val_transforms
from src.models.classifier import TriageClassifier
from src.models.unetpp import build_unetpp

CLASS_NAMES = {0: "Healthy", 1: "Non-DFU Wound", 2: "DFU (Diabetic Foot Ulcer)"}


def load_and_preprocess(image_path: str) -> tuple[np.ndarray, torch.Tensor]:
    """Load image and prepare for inference."""
    image = cv2.imread(image_path)
    if image is None:
        msg = f"Cannot read image: {image_path}"
        raise FileNotFoundError(msg)

    # Keep original for display
    original = image.copy()

    # Preprocess for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (512, 512))

    transform = get_val_transforms()
    transformed = transform(image=image_resized)
    tensor = transformed["image"].unsqueeze(0)

    return original, tensor


def main() -> None:
    """Run inference on a single image."""
    parser = argparse.ArgumentParser(description="DiaFoot.AI v2 — Predict")
    parser.add_argument("--image", type=str, required=True, help="Path to foot image")
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default="checkpoints/classifier/best_epoch004_1.0000.pt",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=str,
        default="checkpoints/segmentation/best_epoch019_0.6781.pt",
    )
    parser.add_argument("--save-mask", type=str, default=None, help="Save segmentation mask")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # Load image
    original, tensor = load_and_preprocess(args.image)
    tensor = tensor.to(device)

    print(f"\n{'=' * 50}")
    print("DiaFoot.AI v2 — Inference")
    print(f"Image: {args.image}")
    print(f"{'=' * 50}")

    # Step 1: Classification
    classifier_path = Path(args.classifier_checkpoint)
    if classifier_path.exists():
        classifier = TriageClassifier(
            backbone="tf_efficientnetv2_m", num_classes=3, pretrained=False
        )
        ckpt = torch.load(str(classifier_path), map_location="cpu", weights_only=True)
        classifier.load_state_dict(ckpt["model_state_dict"])
        classifier = classifier.to(device).eval()

        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_class = int(probs.argmax())
        confidence = float(probs.max())

        print(f"\n  Classification: {CLASS_NAMES[pred_class]}")
        print(f"  Confidence: {confidence:.1%}")
        for i, name in CLASS_NAMES.items():
            print(f"    {name}: {probs[i]:.1%}")
    else:
        print(f"\n  Classifier checkpoint not found: {classifier_path}")
        pred_class = 2  # Assume DFU for segmentation

    # Step 2: Segmentation (if wound detected)
    segmenter_path = Path(args.segmenter_checkpoint)
    if pred_class in (1, 2) and segmenter_path.exists():
        segmenter = build_unetpp(encoder_name="efficientnet-b4", encoder_weights=None, classes=1)
        ckpt = torch.load(str(segmenter_path), map_location="cpu", weights_only=True)
        segmenter.load_state_dict(ckpt["model_state_dict"])
        segmenter = segmenter.to(device).eval()

        with torch.no_grad():
            seg_logits = segmenter(tensor)
            seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            seg_mask = (seg_prob > 0.5).astype(np.uint8)

        wound_pixels = seg_mask.sum()
        total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
        coverage = wound_pixels / total_pixels * 100
        area_mm2 = wound_pixels * 0.5 * 0.5  # Assuming 0.5mm/pixel

        print("\n  Segmentation:")
        print(f"    Wound detected: {'Yes' if wound_pixels > 0 else 'No'}")
        print(f"    Wound pixels: {wound_pixels:,}")
        print(f"    Coverage: {coverage:.1f}%")
        print(f"    Estimated area: {area_mm2:.1f} mm2")

        if args.save_mask:
            mask_resized = cv2.resize(seg_mask * 255, (original.shape[1], original.shape[0]))
            cv2.imwrite(args.save_mask, mask_resized)
            print(f"    Mask saved to: {args.save_mask}")
    elif pred_class == 0:
        print("\n  Segmentation: Skipped (healthy foot detected)")
    else:
        print(f"\n  Segmenter checkpoint not found: {segmenter_path}")

    print(f"\n{'=' * 50}\n")


if __name__ == "__main__":
    main()
