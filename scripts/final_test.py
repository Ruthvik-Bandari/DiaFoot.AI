#!/usr/bin/env python3
"""
Final test with optimized pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2

from src.inference.optimized_pipeline import load_pipeline


def main():
    # Load pipeline
    pipeline = load_pipeline("outputs/fuseg_simple/best_model.pt")
    
    # Test images
    val_images_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/images")
    val_masks_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/labels")
    
    image_files = sorted(list(val_images_dir.glob("*.png")))[:10]
    
    print(f"\n{'='*60}")
    print(f"{'Image':<15} {'IoU':<10} {'Dice':<10} {'Wound %':<10}")
    print(f"{'='*60}")
    
    total_iou = 0
    total_dice = 0
    
    output_dir = Path("outputs/final_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_files:
        # Load image and ground truth
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask_path = val_masks_dir / img_path.name
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = (gt_mask > 127).astype(np.uint8)
        
        # Predict
        result = pipeline.predict(image)
        pred_mask = result["mask"]
        
        # Calculate metrics
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection
        iou = intersection / (union + 1e-6)
        dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
        
        total_iou += iou
        total_dice += dice
        
        print(f"{img_path.name:<15} {iou:.4f}     {dice:.4f}     {result['wound_percentage']:.2f}%")
        
        # Save visualization
        vis = pipeline.visualize(image, result)
        cv2.imwrite(str(output_dir / f"{img_path.stem}_final.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    n = len(image_files)
    print(f"{'='*60}")
    print(f"{'AVERAGE':<15} {total_iou/n:.4f}     {total_dice/n:.4f}")
    print(f"{'='*60}")
    
    print(f"\n✅ Results saved to: {output_dir}/")
    
    print(f"\n{'='*60}")
    print(f"📊 FINAL MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Training Validation IoU:  0.8493")
    print(f"  Test with Optimizations:  {total_iou/n:.4f}")
    print(f"  Test Dice Score:          {total_dice/n:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
