#!/usr/bin/env python3
"""
Test trained model on sample images and visualize results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import cv2

from src.models.segmentation import SegmentationModel


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model."""
    model = SegmentationModel(
        architecture="unetplusplus",
        encoder_name="efficientnet-b4",
        encoder_weights=None,  # We're loading our own weights
        num_classes=1,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Val IoU: {checkpoint['val_iou']:.4f}")
    print(f"   Val Dice: {checkpoint['val_dice']:.4f}")
    
    return model


def preprocess_image(image: np.ndarray, size: int = 512) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize
    image = cv2.resize(image, (size, size))
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    
    # To tensor: HWC -> CHW
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    return tensor


def predict(model, image: np.ndarray, device: torch.device, threshold: float = 0.5):
    """Run prediction on single image."""
    # Preprocess
    input_tensor = preprocess_image(image).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Threshold to binary mask
    mask = (prob_map > threshold).astype(np.uint8)
    
    return mask, prob_map


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create visualization with mask overlay."""
    # Resize mask to match image
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Create colored overlay (red for wound)
    overlay = image.copy()
    overlay[mask_resized > 0] = [255, 0, 0]  # Red
    
    # Blend
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    # Add contour
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Green contour
    
    return result


def calculate_wound_stats(mask: np.ndarray, prob_map: np.ndarray) -> dict:
    """Calculate wound statistics."""
    wound_pixels = mask.sum()
    total_pixels = mask.size
    
    return {
        "wound_percentage": wound_pixels / total_pixels * 100,
        "wound_pixels": int(wound_pixels),
        "total_pixels": total_pixels,
        "mean_confidence": float(prob_map[mask > 0].mean()) if wound_pixels > 0 else 0,
        "max_confidence": float(prob_map.max()),
    }


def test_on_validation_set(model, device, num_samples: int = 5):
    """Test model on validation images."""
    val_images_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/images")
    val_masks_dir = Path("data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge/validation/labels")
    
    if not val_images_dir.exists():
        print(f"❌ Validation directory not found: {val_images_dir}")
        return
    
    image_files = sorted(list(val_images_dir.glob("*.png")))[:num_samples]
    
    if not image_files:
        image_files = sorted(list(val_images_dir.glob("*.jpg")))[:num_samples]
    
    print(f"\n📊 Testing on {len(image_files)} validation images...\n")
    
    output_dir = Path("outputs/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_iou = 0
    total_dice = 0
    
    for img_path in image_files:
        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load ground truth mask
        mask_path = val_masks_dir / img_path.name
        if mask_path.exists():
            gt_mask = np.array(Image.open(mask_path).convert("L"))
            gt_mask = (gt_mask > 127).astype(np.uint8)
        else:
            gt_mask = None
        
        # Predict
        pred_mask, prob_map = predict(model, image, device)
        
        # Resize prediction to original size
        pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
        
        # Calculate metrics if ground truth available
        if gt_mask is not None:
            # IoU
            intersection = (pred_mask_resized * gt_mask).sum()
            union = pred_mask_resized.sum() + gt_mask.sum() - intersection
            iou = intersection / (union + 1e-6)
            
            # Dice
            dice = (2 * intersection) / (pred_mask_resized.sum() + gt_mask.sum() + 1e-6)
            
            total_iou += iou
            total_dice += dice
            
            metrics_str = f"IoU: {iou:.4f}, Dice: {dice:.4f}"
        else:
            metrics_str = "No ground truth"
        
        # Calculate wound stats
        stats = calculate_wound_stats(pred_mask, prob_map)
        
        # Create visualization
        overlay = create_overlay(image, pred_mask_resized, alpha=0.4)
        
        # Save results
        output_path = output_dir / f"{img_path.stem}_result.png"
        
        # Create side-by-side comparison
        h, w = image.shape[:2]
        canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original
        canvas[:, :w] = image
        
        # Prediction overlay
        canvas[:, w:w*2] = overlay
        
        # Mask visualization
        mask_vis = cv2.resize(prob_map, (w, h))
        mask_vis = (mask_vis * 255).astype(np.uint8)
        mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        canvas[:, w*2:] = mask_vis
        
        # Save
        cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        print(f"📸 {img_path.name}")
        print(f"   {metrics_str}")
        print(f"   Wound: {stats['wound_percentage']:.2f}% of image")
        print(f"   Confidence: {stats['mean_confidence']:.2f}")
        print(f"   Saved: {output_path}\n")
    
    if total_iou > 0:
        print(f"═══════════════════════════════════════")
        print(f"📊 Average IoU:  {total_iou / len(image_files):.4f}")
        print(f"📊 Average Dice: {total_dice / len(image_files):.4f}")
        print(f"═══════════════════════════════════════")
    
    print(f"\n✅ Results saved to: {output_dir}/")


def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model("outputs/fuseg_simple/best_model.pt", device)
    
    # Test on validation set
    test_on_validation_set(model, device, num_samples=10)


if __name__ == "__main__":
    main()
