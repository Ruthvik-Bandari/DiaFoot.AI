"""DiaFoot.AI v2 — Generate Visualization Figures.

Creates overlay images showing predictions vs ground truth for the report.

Usage:
    python scripts/visualize_results.py \
        --checkpoint checkpoints/dinov2_segmenter/best.pt \
        --num-images 10 --device cpu
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
from src.data.torch_dataset import DFUDataset
from src.evaluation.metrics import dice_score


def create_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create visualization with GT (green) and prediction (red) overlays.

    Args:
        image: Original image (H, W, 3) RGB.
        gt_mask: Ground truth binary mask (H, W).
        pred_mask: Predicted binary mask (H, W).
        alpha: Overlay transparency.

    Returns:
        Visualization image (H, W, 3).
    """
    vis = image.copy()

    # Green overlay for ground truth
    gt_overlay = np.zeros_like(vis)
    gt_overlay[gt_mask > 0] = [0, 255, 0]
    vis = np.where(
        gt_mask[:, :, None] > 0,
        (vis * (1 - alpha) + gt_overlay * alpha).astype(np.uint8),
        vis,
    )

    # Red overlay for prediction
    pred_overlay = np.zeros_like(vis)
    pred_overlay[pred_mask > 0] = [255, 0, 0]
    vis = np.where(
        pred_mask[:, :, None] > 0,
        (vis * (1 - alpha) + pred_overlay * alpha).astype(np.uint8),
        vis,
    )

    # Yellow where they overlap (green + red = yellow)
    overlap = (gt_mask > 0) & (pred_mask > 0)
    overlap_color = np.zeros_like(vis)
    overlap_color[overlap] = [255, 255, 0]
    vis = np.where(
        overlap[:, :, None],
        (image * (1 - alpha) + overlap_color * alpha).astype(np.uint8),
        vis,
    )

    return vis


def create_comparison_grid(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    dice: float,
    sample_idx: int,
) -> np.ndarray:
    """Create a 1x4 comparison grid: Original | GT | Pred | Overlay.

    Args:
        image: Original image (H, W, 3).
        gt_mask: Ground truth mask (H, W).
        pred_mask: Predicted mask (H, W).
        dice: Dice score for this sample.
        sample_idx: Sample index for labeling.

    Returns:
        Grid image.
    """
    h, w = image.shape[:2]

    # Convert masks to 3-channel for display
    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt_mask > 0] = [0, 255, 0]

    pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_vis[pred_mask > 0] = [255, 0, 0]

    overlay = create_overlay(image, gt_mask, pred_mask)

    # Concatenate horizontally
    grid = np.concatenate([image, gt_vis, pred_vis, overlay], axis=1)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Original", (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(grid, "GT Mask", (w + 10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(grid, "Prediction", (2 * w + 10, 25), font, 0.6, (0, 0, 255), 2)
    cv2.putText(
        grid,
        f"Overlay (Dice={dice:.3f})",
        (3 * w + 10, 25),
        font,
        0.5,
        (255, 255, 0),
        2,
    )

    return grid


def main() -> None:
    """Generate visualization figures."""
    parser = argparse.ArgumentParser(description="Visualize Results")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("visualize")

    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from src.models.unetpp import build_unetpp

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
        return_metadata=True,
    )

    # Generate visualizations
    all_grids = []

    for idx in range(min(args.num_images, len(test_ds))):
        sample = test_ds[idx]
        image_tensor = sample["image"].unsqueeze(0).to(dev)
        gt_mask = sample["mask"].numpy()
        label = sample["label"]

        # Skip healthy images (no wound to visualize)
        if label == 0 and gt_mask.sum() == 0:
            continue

        # Predict
        with torch.no_grad():
            logits = model(image_tensor)
            pred = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        # Denormalize image for visualization
        img = sample["image"].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = ((img * std + mean) * 255).clip(0, 255).astype(np.uint8)

        # Compute metrics
        dice = dice_score(pred, gt_mask)

        # Create grid
        grid = create_comparison_grid(img, gt_mask, pred, dice, idx)
        all_grids.append(grid)

        # Save individual
        cv2.imwrite(
            str(output_dir / f"sample_{idx:03d}_dice{dice:.3f}.png"),
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
        )

    # Create combined figure (stack vertically)
    if all_grids:
        combined = np.concatenate(all_grids, axis=0)
        cv2.imwrite(
            str(output_dir / "combined_results.png"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
        )
        logger.info("Saved %d visualizations to %s", len(all_grids), output_dir)
    else:
        logger.warning("No images with wounds found for visualization")


if __name__ == "__main__":
    main()
