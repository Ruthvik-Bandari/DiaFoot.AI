"""
Preprocessing Module
=====================

Image preprocessing utilities for wound images.
Includes normalization, resizing, and quality assessment.

Author: Ruthvik
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from pathlib import Path


def preprocess_image(
    image: np.ndarray,
    target_size: int = 512,
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Preprocess a single image for model input.
    
    Args:
        image: Input image (H, W, C) in RGB format
        target_size: Target size for the image
        normalize: Whether to apply ImageNet normalization
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Preprocessed image ready for model input
    """
    # Resize with aspect ratio preservation
    image = resize_with_padding(image, target_size)
    
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    if normalize:
        image = normalize_image(image, mean, std)
    
    return image


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Apply ImageNet normalization to image.
    
    Args:
        image: Input image (H, W, C) with values in [0, 1]
        mean: Normalization mean per channel
        std: Normalization std per channel
    
    Returns:
        Normalized image
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    return (image - mean) / std


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Reverse ImageNet normalization.
    
    Args:
        image: Normalized image
        mean: Normalization mean per channel
        std: Normalization std per channel
    
    Returns:
        Denormalized image with values in [0, 1]
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = image * std + mean
    return np.clip(image, 0, 1)


def resize_with_padding(
    image: np.ndarray,
    target_size: int,
    padding_value: int = 0,
) -> np.ndarray:
    """
    Resize image while preserving aspect ratio with padding.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target size (square output)
        padding_value: Value to use for padding (0-255)
    
    Returns:
        Resized and padded image of shape (target_size, target_size, C)
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded output
    if len(image.shape) == 3:
        padded = np.full((target_size, target_size, image.shape[2]), 
                         padding_value, dtype=image.dtype)
    else:
        padded = np.full((target_size, target_size), padding_value, dtype=image.dtype)
    
    # Center the resized image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


def remove_padding(
    image: np.ndarray,
    original_size: Tuple[int, int],
    padded_size: int,
) -> np.ndarray:
    """
    Remove padding and restore original aspect ratio.
    
    Args:
        image: Padded image
        original_size: Original (H, W) before padding
        padded_size: Size of the padded image
    
    Returns:
        Image with padding removed and resized to original
    """
    orig_h, orig_w = original_size
    
    # Calculate scale that was used
    scale = padded_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    # Calculate padding offsets
    y_offset = (padded_size - new_h) // 2
    x_offset = (padded_size - new_w) // 2
    
    # Extract the non-padded region
    cropped = image[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    
    # Resize back to original
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    return restored


def assess_image_quality(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    brightness_range: Tuple[float, float] = (30, 220),
) -> dict:
    """
    Assess image quality for wound analysis.
    
    Args:
        image: Input image in RGB format
        blur_threshold: Laplacian variance threshold for blur detection
        brightness_range: Acceptable brightness range
    
    Returns:
        Dictionary with quality assessment results
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Check for blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < blur_threshold
    
    # Check brightness
    mean_brightness = gray.mean()
    is_too_dark = mean_brightness < brightness_range[0]
    is_too_bright = mean_brightness > brightness_range[1]
    
    # Check contrast
    contrast = gray.std()
    is_low_contrast = contrast < 30
    
    # Overall quality score (0-100)
    quality_score = 100
    if is_blurry:
        quality_score -= 30
    if is_too_dark or is_too_bright:
        quality_score -= 25
    if is_low_contrast:
        quality_score -= 20
    
    return {
        "quality_score": max(0, quality_score),
        "is_acceptable": quality_score >= 50,
        "is_blurry": is_blurry,
        "blur_score": laplacian_var,
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
        "mean_brightness": mean_brightness,
        "is_low_contrast": is_low_contrast,
        "contrast": contrast,
        "issues": [
            issue for issue, present in [
                ("blurry", is_blurry),
                ("too_dark", is_too_dark),
                ("too_bright", is_too_bright),
                ("low_contrast", is_low_contrast),
            ] if present
        ],
    }


def enhance_wound_image(
    image: np.ndarray,
    clahe_clip: float = 2.0,
    sharpen: bool = True,
) -> np.ndarray:
    """
    Enhance wound image for better visibility.
    
    Args:
        image: Input image in RGB format
        clahe_clip: CLAHE clip limit
        sharpen: Whether to apply sharpening
    
    Returns:
        Enhanced image
    """
    # Convert to LAB for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge channels
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Optional sharpening
    if sharpen:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def extract_wound_roi(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract wound region of interest based on mask.
    
    Args:
        image: Input image
        mask: Binary wound mask
        padding: Padding around the ROI
    
    Returns:
        Tuple of (cropped_image, cropped_mask, bbox)
    """
    # Find bounding box of the mask
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        # No wound detected, return original
        h, w = image.shape[:2]
        return image, mask, (0, 0, w, h)
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Add padding
    h, w = image.shape[:2]
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding)
    
    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    
    return cropped_image, cropped_mask, (x_min, y_min, x_max, y_max)


def compute_wound_metrics(mask: np.ndarray, pixel_to_mm: float = 0.1) -> dict:
    """
    Compute wound metrics from segmentation mask.
    
    Args:
        mask: Binary wound mask
        pixel_to_mm: Conversion factor from pixels to millimeters
    
    Returns:
        Dictionary with wound metrics
    """
    # Find contours
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            "area_pixels": 0,
            "area_mm2": 0,
            "perimeter_pixels": 0,
            "perimeter_mm": 0,
            "circularity": 0,
            "num_regions": 0,
        }
    
    # Get the largest contour (main wound)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Calculate metrics
    area_pixels = cv2.contourArea(main_contour)
    perimeter_pixels = cv2.arcLength(main_contour, True)
    
    # Circularity (1 = perfect circle)
    circularity = 0
    if perimeter_pixels > 0:
        circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(main_contour)
    
    return {
        "area_pixels": area_pixels,
        "area_mm2": area_pixels * (pixel_to_mm ** 2),
        "perimeter_pixels": perimeter_pixels,
        "perimeter_mm": perimeter_pixels * pixel_to_mm,
        "circularity": circularity,
        "num_regions": len(contours),
        "bounding_box": {"x": x, "y": y, "width": w, "height": h},
        "aspect_ratio": w / h if h > 0 else 0,
    }
