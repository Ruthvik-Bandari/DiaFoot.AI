#!/usr/bin/env python3
"""
Compare basic inference vs enhanced pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import cv2

from src.models.segmentation import SegmentationModel
from src.inference.enhanced_pipeline import create_pipeline


def load_basic_model(checkpoint_path: str, device: torch.device):
    """Load model for basic inference."""
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


def basic_predict(model, image, device, size=512):
    """Basic prediction without enhancements."""
    # Preprocess
    resized = cv2.resize(image, (size, size))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Predict
    with torch.no_grad():
        output = model(tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Resize to original
    prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]))
    mask = (prob_map > 0.5).astype(np.uint8)
    
    return mask, prob_map


def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU and Dice."""
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    
    return iou, dice


def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}\n")
    
    # Load models
    model_path = "outputs/fuseg_simple/best_model.pt"
    
    print("Loading basic model...")
    basic_model = load_basic_model(model_path, device)
    
    print("Loading enhanced pipeline...")
    enhanced_pipeline = create_pipeline(model_path)
    
    # Test images
    val_images_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/images")
    val_masks_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/labels")
    
    image_files = sorted(list(val_images_dir.glob("*.png")))[:10]
    
    print(f"\n{'='*70}")
    print(f"{'Image':<12} {'Basic IoU':<12} {'Enhanced IoU':<14} {'Improvement':<12}")
    print(f"{'='*70}")
    
    basic_total_iou = 0
    enhanced_total_iou = 0
    
    output_dir = Path("outputs/enhanced_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_files:
        # Load image and ground truth
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask_path = val_masks_dir / img_path.name
        gt_mask = np.array(Image.open(mask_path).convert("L"))
        gt_mask = (gt_mask > 127).astype(np.uint8)
        
        # Basic prediction
        basic_mask, _ = basic_predict(basic_model, image, device)
        basic_iou, basic_dice = calculate_metrics(basic_mask, gt_mask)
        
        # Enhanced prediction
        enhanced_result = enhanced_pipeline.predict(image)
        enhanced_mask = enhanced_result["mask"]
        enhanced_iou, enhanced_dice = calculate_metrics(enhanced_mask, gt_mask)
        
        # Calculate improvement
        improvement = enhanced_iou - basic_iou
        improvement_str = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        
        print(f"{img_path.name:<12} {basic_iou:.4f}       {enhanced_iou:.4f}         {improvement_str}")
        
        basic_total_iou += basic_iou
        enhanced_total_iou += enhanced_iou
        
        # Save visualization
        vis = enhanced_pipeline.visualize(image, enhanced_result)
        cv2.imwrite(
            str(output_dir / f"{img_path.stem}_enhanced.png"),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )
    
    # Averages
    n = len(image_files)
    avg_improvement = (enhanced_total_iou - basic_total_iou) / n
    
    print(f"{'='*70}")
    print(f"{'AVERAGE':<12} {basic_total_iou/n:.4f}       {enhanced_total_iou/n:.4f}         +{avg_improvement:.4f}")
    print(f"{'='*70}")
    
    print(f"\n✅ Enhanced results saved to: {output_dir}/")
    print(f"\n📊 Summary:")
    print(f"   Basic Average IoU:    {basic_total_iou/n:.4f}")
    print(f"   Enhanced Average IoU: {enhanced_total_iou/n:.4f}")
    print(f"   Improvement:          +{avg_improvement:.4f} (+{avg_improvement/basic_total_iou*n*100:.1f}%)")


if __name__ == "__main__":
    main()
