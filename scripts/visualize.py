#!/usr/bin/env python3
"""
DiaFootAI Visualization & Demo Tools
=====================================

Creates stunning visualizations for presentations:
1. Side-by-side comparisons (image | prediction | ground truth)
2. Overlay visualizations (colored mask on image)
3. Boundary comparison (predicted vs ground truth edges)
4. Batch processing for multiple images
5. Interactive demo mode

Author: Ruthvik
Date: January 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Professional color schemes for medical imaging
COLORS = {
    'wound_red': (220, 60, 60),      # Red for wound area
    'wound_blue': (65, 105, 225),    # Royal blue
    'wound_green': (50, 205, 50),    # Lime green
    'boundary_yellow': (255, 215, 0), # Gold for boundaries
    'correct_green': (0, 255, 0),    # Green for correct predictions
    'false_positive': (255, 0, 0),   # Red for false positives
    'false_negative': (0, 0, 255),   # Blue for false negatives
}


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    checkpoint_path: str,
    encoder: str = 'efficientnet-b4',
    architecture: str = 'unetplusplus',
    device: str = 'cpu'
) -> nn.Module:
    """Load trained model from checkpoint."""
    
    if architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    else:
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


# =============================================================================
# PREDICTION
# =============================================================================

class Predictor:
    """Simple predictor for visualization."""
    
    def __init__(self, model: nn.Module, device: str, image_size: int = 512):
        self.model = model
        self.device = device
        self.image_size = image_size
        
        self.preprocess = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation mask.
        
        Returns:
            Tuple of (probability_map, binary_mask) at model size (512x512)
        """
        result = self.preprocess(image=image)
        tensor = result['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        mask = (prob > 0.5).astype(np.uint8)
        
        return prob, mask


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = COLORS['wound_red'],
    alpha: float = 0.5
) -> np.ndarray:
    """Create colored overlay of mask on image."""
    
    # Ensure image is right size
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = image.copy()
    mask_bool = mask > 0
    
    for i, c in enumerate(color):
        overlay[:, :, i] = np.where(
            mask_bool,
            overlay[:, :, i] * (1 - alpha) + c * alpha,
            overlay[:, :, i]
        )
    
    return overlay.astype(np.uint8)


def create_boundary_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_color: Tuple[int, int, int] = COLORS['wound_red'],
    gt_color: Tuple[int, int, int] = COLORS['boundary_yellow'],
    thickness: int = 2
) -> np.ndarray:
    """Create boundary comparison visualization."""
    
    # Resize masks if needed
    h, w = image.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Find contours
    pred_contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw boundaries
    result = image.copy()
    cv2.drawContours(result, gt_contours, -1, gt_color, thickness)
    cv2.drawContours(result, pred_contours, -1, pred_color, thickness)
    
    return result


def create_error_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create error visualization showing:
    - Green: Correct predictions (true positives)
    - Red: False positives
    - Blue: False negatives
    """
    
    h, w = gt_mask.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    # Create RGB error map
    if image is not None:
        error_map = (image * 0.3).astype(np.uint8)  # Dimmed background
    else:
        error_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # True positives (green)
    tp = np.logical_and(pred_bool, gt_bool)
    error_map[tp] = COLORS['correct_green']
    
    # False positives (red)
    fp = np.logical_and(pred_bool, ~gt_bool)
    error_map[fp] = COLORS['false_positive']
    
    # False negatives (blue)
    fn = np.logical_and(~pred_bool, gt_bool)
    error_map[fn] = COLORS['false_negative']
    
    return error_map


def create_comparison_figure(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    pred_prob: Optional[np.ndarray] = None,
    title: str = "",
    figsize: Tuple[int, int] = (20, 5)
) -> plt.Figure:
    """
    Create comprehensive comparison figure for presentations.
    
    Shows: Original | Prediction Overlay | Probability Map | Error Map (if GT available)
    """
    
    n_cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Resize for display
    h, w = image.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        pred_mask_display = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        pred_mask_display = pred_mask
    
    if pred_prob is not None and pred_prob.shape[:2] != (h, w):
        pred_prob_display = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        pred_prob_display = pred_prob
    
    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Prediction overlay
    overlay = create_overlay(image, pred_mask_display, COLORS['wound_red'], alpha=0.5)
    axes[1].imshow(overlay)
    axes[1].set_title("Predicted Wound Area", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Probability map
    if pred_prob_display is not None:
        # Custom colormap: black -> blue -> red -> yellow
        colors = ['black', 'darkblue', 'blue', 'red', 'yellow']
        cmap = LinearSegmentedColormap.from_list('prob_cmap', colors)
        
        im = axes[2].imshow(pred_prob_display, cmap=cmap, vmin=0, vmax=1)
        axes[2].set_title("Confidence Map", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].imshow(pred_mask_display, cmap='Reds')
        axes[2].set_title("Prediction Mask", fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    # 4. Error map (if ground truth available)
    if gt_mask is not None:
        if gt_mask.shape[:2] != (h, w):
            gt_mask_display = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            gt_mask_display = gt_mask
        
        error_map = create_error_map(pred_mask_display, gt_mask_display, image)
        axes[3].imshow(error_map)
        axes[3].set_title("Error Analysis\n(🟢 Correct  🔴 FP  🔵 FN)", fontsize=14, fontweight='bold')
        axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def create_metrics_badge(
    iou: float,
    dice: float,
    precision: float,
    recall: float
) -> np.ndarray:
    """Create a metrics badge image for overlaying on visualizations."""
    
    # Create badge
    badge_h, badge_w = 120, 200
    badge = np.ones((badge_h, badge_w, 3), dtype=np.uint8) * 40  # Dark background
    
    # Add border
    cv2.rectangle(badge, (0, 0), (badge_w-1, badge_h-1), (100, 100, 100), 2)
    
    # Add metrics text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    
    cv2.putText(badge, f"IoU: {iou:.1%}", (10, 25), font, font_scale, color, 1)
    cv2.putText(badge, f"Dice: {dice:.1%}", (10, 50), font, font_scale, color, 1)
    cv2.putText(badge, f"Precision: {precision:.1%}", (10, 75), font, font_scale, color, 1)
    cv2.putText(badge, f"Recall: {recall:.1%}", (10, 100), font, font_scale, color, 1)
    
    return badge


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_batch(
    predictor: Predictor,
    image_dir: Path,
    mask_dir: Optional[Path],
    output_dir: Path,
    num_samples: int = 10
) -> None:
    """Process multiple images and save visualizations."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    if num_samples > 0:
        # Select evenly spaced samples
        indices = np.linspace(0, len(image_paths)-1, num_samples, dtype=int)
        image_paths = [image_paths[i] for i in indices]
    
    print(f"\n🖼️  Processing {len(image_paths)} images...")
    
    for img_path in image_paths:
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth if available
        gt_mask = None
        if mask_dir:
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 127).astype(np.uint8)
        
        # Predict
        prob, pred_mask = predictor.predict(image)
        
        # Create visualization
        fig = create_comparison_figure(
            image=image,
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            pred_prob=prob,
            title=img_path.stem
        )
        
        # Save
        output_path = output_dir / f"{img_path.stem}_viz.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"   ✅ Saved: {output_path.name}")
    
    print(f"\n📁 All visualizations saved to: {output_dir}")


# =============================================================================
# DEMO MODE
# =============================================================================

def run_demo(predictor: Predictor, image_path: str) -> None:
    """Run interactive demo on a single image."""
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"\n🔬 Processing: {image_path}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Predict
    prob, pred_mask = predictor.predict(image)
    
    # Create visualization
    fig = create_comparison_figure(
        image=image,
        pred_mask=pred_mask,
        gt_mask=None,
        pred_prob=prob,
        title="DiaFootAI Wound Segmentation Demo"
    )
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DiaFootAI Visualization Tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    batch_parser.add_argument('--encoder', default='efficientnet-b4', help='Encoder name')
    batch_parser.add_argument('--image-dir', required=True, help='Directory with images')
    batch_parser.add_argument('--mask-dir', help='Directory with ground truth masks')
    batch_parser.add_argument('--output-dir', default='outputs/visualizations', help='Output directory')
    batch_parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo on single image')
    demo_parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    demo_parser.add_argument('--encoder', default='efficientnet-b4', help='Encoder name')
    demo_parser.add_argument('--image', required=True, help='Image path')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"🖥️  Device: {device}")
    
    # Load model
    print(f"📦 Loading model: {args.encoder}")
    model = load_model(args.checkpoint, args.encoder, device=device)
    predictor = Predictor(model, device)
    
    if args.command == 'batch':
        process_batch(
            predictor=predictor,
            image_dir=Path(args.image_dir),
            mask_dir=Path(args.mask_dir) if args.mask_dir else None,
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples
        )
    
    elif args.command == 'demo':
        run_demo(predictor, args.image)


if __name__ == "__main__":
    main()
