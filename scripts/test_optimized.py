#!/usr/bin/env python3
"""
Test different enhancement configurations to find optimal settings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import cv2

from src.models.segmentation import SegmentationModel
from src.inference.enhanced_pipeline import (
    EnhancedWoundSegmentationPipeline,
    CLAHEPreprocessor,
    PostProcessor,
    TestTimeAugmentation,
)


def load_model(checkpoint_path, device):
    model = SegmentationModel(
        architecture="unetplusplus",
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        num_classes=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def calculate_metrics(pred_mask, gt_mask):
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    return iou, dice


def test_configuration(model, images, gt_masks, device, config_name, 
                       use_clahe=False, use_tta=False, use_post=False,
                       tta_flips_only=False, min_region_size=100):
    """Test a specific configuration."""
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    clahe = CLAHEPreprocessor() if use_clahe else None
    post = PostProcessor(min_size=min_region_size) if use_post else None
    tta = TestTimeAugmentation(use_flip=True, use_rotate=not tta_flips_only) if use_tta else None
    
    total_iou = 0
    
    for image, gt_mask in zip(images, gt_masks):
        # Preprocess
        if use_clahe:
            processed = clahe.apply(image)
        else:
            processed = image
        
        # Resize and normalize
        resized = cv2.resize(processed, (512, 512))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - mean) / std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Predict
        with torch.no_grad():
            if use_tta:
                augmented = tta.augment(processed)
                preds = []
                for aug_img in augmented:
                    aug_resized = cv2.resize(aug_img, (512, 512))
                    aug_norm = aug_resized.astype(np.float32) / 255.0
                    aug_norm = (aug_norm - mean) / std
                    aug_tensor = torch.from_numpy(aug_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    out = model(aug_tensor)
                    preds.append(torch.sigmoid(out).squeeze().cpu().numpy())
                prob_map = tta.deaugment(preds)
            else:
                output = model(tensor)
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Resize to original
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))
        mask = (prob_map > 0.5).astype(np.uint8)
        
        # Post-process
        if use_post:
            mask = post.apply(mask)
        
        iou, _ = calculate_metrics(mask, gt_mask)
        total_iou += iou
    
    avg_iou = total_iou / len(images)
    return avg_iou


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load model
    model = load_model("outputs/fuseg_simple/best_model.pt", device)
    
    # Load test images
    val_images_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/images")
    val_masks_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/labels")
    
    image_files = sorted(list(val_images_dir.glob("*.png")))[:10]
    
    images = []
    gt_masks = []
    for img_path in image_files:
        images.append(np.array(Image.open(img_path).convert("RGB")))
        mask_path = val_masks_dir / img_path.name
        gt = np.array(Image.open(mask_path).convert("L"))
        gt_masks.append((gt > 127).astype(np.uint8))
    
    print("Testing different configurations...\n")
    print(f"{'Configuration':<45} {'Avg IoU':<10} {'vs Basic':<10}")
    print("=" * 65)
    
    # Test configurations
    configs = [
        ("1. Basic (no enhancements)", dict(use_clahe=False, use_tta=False, use_post=False)),
        ("2. CLAHE only", dict(use_clahe=True, use_tta=False, use_post=False)),
        ("3. TTA only (flips + rotations)", dict(use_clahe=False, use_tta=True, use_post=False)),
        ("4. TTA only (flips only)", dict(use_clahe=False, use_tta=True, use_post=False, tta_flips_only=True)),
        ("5. Post-processing only", dict(use_clahe=False, use_tta=False, use_post=True)),
        ("6. Post-processing (smaller min)", dict(use_clahe=False, use_tta=False, use_post=True, min_region_size=50)),
        ("7. CLAHE + TTA (flips only)", dict(use_clahe=True, use_tta=True, use_post=False, tta_flips_only=True)),
        ("8. CLAHE + Post-processing", dict(use_clahe=True, use_tta=False, use_post=True)),
        ("9. TTA (flips) + Post-processing", dict(use_clahe=False, use_tta=True, use_post=True, tta_flips_only=True)),
        ("10. ALL (flips only, small min)", dict(use_clahe=True, use_tta=True, use_post=True, tta_flips_only=True, min_region_size=50)),
    ]
    
    baseline_iou = None
    best_config = None
    best_iou = 0
    
    for name, kwargs in configs:
        iou = test_configuration(model, images, gt_masks, device, name, **kwargs)
        
        if baseline_iou is None:
            baseline_iou = iou
            diff = "baseline"
        else:
            diff = f"+{iou - baseline_iou:.4f}" if iou > baseline_iou else f"{iou - baseline_iou:.4f}"
        
        print(f"{name:<45} {iou:.4f}     {diff}")
        
        if iou > best_iou:
            best_iou = iou
            best_config = name
    
    print("=" * 65)
    print(f"\n🏆 Best Configuration: {best_config}")
    print(f"   IoU: {best_iou:.4f} (vs {baseline_iou:.4f} baseline)")
    print(f"   Improvement: +{best_iou - baseline_iou:.4f} (+{(best_iou - baseline_iou) / baseline_iou * 100:.1f}%)")


if __name__ == "__main__":
    main()
