"""
Visualization Utilities
========================

Functions for visualizing images, masks, and predictions.
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


# Default color palette for tissue types
TISSUE_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Granulation - Red
    2: (255, 255, 0),    # Slough - Yellow
    3: (64, 64, 64),     # Necrotic - Dark Gray
    4: (255, 192, 203),  # Epithelial - Pink
    5: (255, 165, 0),    # Periwound - Orange
}

TISSUE_NAMES = {
    0: "Background",
    1: "Granulation",
    2: "Slough",
    3: "Necrotic",
    4: "Epithelial",
    5: "Periwound",
}


def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy image.
    
    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W)
        denormalize: Whether to denormalize using ImageNet stats
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Image as numpy array (H, W, C) in range [0, 255]
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU and convert to numpy
    image = tensor.detach().cpu().numpy()
    
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    
    # Denormalize
    if denormalize:
        mean = np.array(mean)
        std = np.array(std)
        image = image * std + mean
    
    # Clip and convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def mask_to_rgb(
    mask: Union[np.ndarray, torch.Tensor],
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Convert a segmentation mask to RGB image.
    
    Args:
        mask: Segmentation mask (H, W) with class indices
        colors: Dictionary mapping class index to RGB color
        num_classes: Number of classes (for auto-generating colors)
        
    Returns:
        RGB image (H, W, 3)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Squeeze extra dimensions
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    if colors is None:
        colors = TISSUE_COLORS
    
    # Create RGB image
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in colors.items():
        rgb[mask == class_idx] = color
    
    return rgb


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Overlay a segmentation mask on an image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Segmentation mask (H, W)
        alpha: Transparency of overlay (0-1)
        colors: Color mapping for mask
        
    Returns:
        Image with mask overlay
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert mask to RGB
    mask_rgb = mask_to_rgb(mask, colors)
    
    # Create overlay
    overlay = image.copy()
    
    # Only overlay where mask is non-zero
    mask_indices = mask > 0
    if mask_indices.any():
        overlay[mask_indices] = cv2.addWeighted(
            image[mask_indices], 1 - alpha,
            mask_rgb[mask_indices], alpha,
            0
        )
    
    return overlay


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw contours of a binary mask on an image.
    
    Args:
        image: Original image
        mask: Binary mask
        color: Contour color (RGB)
        thickness: Contour thickness
        
    Returns:
        Image with contours drawn
    """
    # Convert to BGR for OpenCV
    color_bgr = (color[2], color[1], color[0])
    
    # Ensure mask is binary uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    result = image.copy()
    cv2.drawContours(result, contours, -1, color_bgr, thickness)
    
    return result


def create_comparison_image(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 3,
    padding: int = 10,
    title_height: int = 30,
    font_scale: float = 0.7,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a comparison grid of images.
    
    Args:
        images: List of images to display
        titles: Optional titles for each image
        cols: Number of columns
        padding: Padding between images
        title_height: Height reserved for titles
        font_scale: Font scale for titles
        background_color: Background color
        
    Returns:
        Combined comparison image
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # Get max image size
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    # Calculate grid size
    cell_h = max_h + title_height + padding
    cell_w = max_w + padding
    grid_h = rows * cell_h + padding
    grid_w = cols * cell_w + padding
    
    # Create background
    grid = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)
    
    # Place images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        y = row * cell_h + padding + title_height
        x = col * cell_w + padding
        
        # Ensure image is 3-channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Center image in cell
        h, w = img.shape[:2]
        y_offset = (max_h - h) // 2
        x_offset = (max_w - w) // 2
        
        grid[y + y_offset:y + y_offset + h, x + x_offset:x + x_offset + w] = img
        
        # Add title
        if titles and i < len(titles):
            cv2.putText(
                grid, titles[i],
                (x + 5, row * cell_h + padding + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), 2
            )
    
    return grid


def visualize_augmentations(
    image: np.ndarray,
    augmentation,
    n_samples: int = 9
) -> np.ndarray:
    """
    Visualize augmentation effects on an image.
    
    Args:
        image: Original image
        augmentation: Albumentations augmentation pipeline
        n_samples: Number of augmented samples to show
        
    Returns:
        Grid of augmented images
    """
    images = [image]
    titles = ["Original"]
    
    for i in range(n_samples - 1):
        augmented = augmentation(image=image)
        images.append(augmented["image"])
        titles.append(f"Aug {i + 1}")
    
    return create_comparison_image(images, titles, cols=3)


def add_legend(
    image: np.ndarray,
    labels: Dict[str, Tuple[int, int, int]],
    position: str = "bottom",
    padding: int = 10,
    legend_height: int = 30
) -> np.ndarray:
    """
    Add a color legend to an image.
    
    Args:
        image: Input image
        labels: Dictionary mapping label names to colors
        position: Legend position ('top' or 'bottom')
        padding: Padding around legend
        legend_height: Height of legend bar
        
    Returns:
        Image with legend
    """
    h, w = image.shape[:2]
    
    # Create legend bar
    legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255
    
    # Calculate item width
    n_items = len(labels)
    item_width = w // n_items
    
    for i, (name, color) in enumerate(labels.items()):
        x_start = i * item_width
        x_end = x_start + 20
        
        # Color box
        legend[5:25, x_start + 5:x_end + 5] = color
        
        # Label text
        cv2.putText(
            legend, name,
            (x_end + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 0), 1
        )
    
    # Combine image and legend
    if position == "top":
        result = np.vstack([legend, image])
    else:
        result = np.vstack([image, legend])
    
    return result


def plot_wound_assessment(
    image: np.ndarray,
    mask: np.ndarray,
    tissue_percentages: Dict[str, float],
    infection_score: float,
    area_cm2: Optional[float] = None
) -> np.ndarray:
    """
    Create a comprehensive wound assessment visualization.
    
    Args:
        image: Original wound image
        mask: Tissue segmentation mask
        tissue_percentages: Dictionary of tissue type percentages
        infection_score: Infection probability (0-1)
        area_cm2: Wound area in cm² (optional)
        
    Returns:
        Assessment visualization image
    """
    h, w = image.shape[:2]
    
    # Create overlay
    overlay = overlay_mask(image, mask, alpha=0.4)
    
    # Add contours
    binary_mask = (mask > 0).astype(np.uint8)
    overlay = draw_contours(overlay, binary_mask, color=(0, 255, 0), thickness=3)
    
    # Create info panel
    panel_width = 300
    panel = np.ones((h, panel_width, 3), dtype=np.uint8) * 255
    
    y_pos = 30
    cv2.putText(panel, "Wound Assessment", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y_pos += 40
    
    # Area
    if area_cm2 is not None:
        cv2.putText(panel, f"Area: {area_cm2:.2f} cm2", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 30
    
    # Infection score
    color = (0, 0, 255) if infection_score > 0.5 else (0, 128, 0)
    cv2.putText(panel, f"Infection Risk: {infection_score:.1%}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    y_pos += 40
    
    # Tissue percentages
    cv2.putText(panel, "Tissue Composition:", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_pos += 25
    
    for tissue, percentage in tissue_percentages.items():
        if percentage > 0:
            cv2.putText(panel, f"  {tissue}: {percentage:.1f}%", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y_pos += 20
    
    # Combine
    result = np.hstack([overlay, panel])
    
    return result
