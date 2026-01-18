"""
I/O Utilities
==============

Functions for loading and saving images, masks, and data files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(
    path: Union[str, Path],
    mode: str = "RGB",
    size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        mode: Color mode ('RGB', 'BGR', 'GRAY')
        size: Optional resize dimensions (width, height)
        
    Returns:
        Image as numpy array (H, W, C) for color or (H, W) for grayscale
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    # Load image
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    # Convert color mode
    if mode == "RGB":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif mode == "GRAY":
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 'BGR' is default OpenCV format, no conversion needed
    
    # Resize if specified
    if size is not None:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    mode: str = "RGB"
) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image as numpy array
        path: Output path
        mode: Color mode of input ('RGB', 'BGR', 'GRAY')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to BGR for OpenCV
    if mode == "RGB" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(path), image)


def load_mask(
    path: Union[str, Path],
    size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load a segmentation mask from file.
    
    Args:
        path: Path to mask file
        size: Optional resize dimensions (width, height)
        
    Returns:
        Mask as numpy array (H, W)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    
    # Load as grayscale
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    
    # Resize if specified
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    
    return mask


def save_mask(
    mask: np.ndarray,
    path: Union[str, Path]
) -> None:
    """
    Save a segmentation mask to file.
    
    Args:
        mask: Mask as numpy array (H, W)
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure mask is uint8
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    cv2.imwrite(str(path), mask)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Output path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def list_images(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of valid extensions (default: common image formats)
        recursive: Whether to search recursively
        
    Returns:
        List of image paths
    """
    directory = Path(directory)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Normalize extensions
    extensions = [ext.lower() for ext in extensions]
    
    images = []
    
    if recursive:
        for ext in extensions:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(set(images))


def get_image_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about an image file.
    
    Args:
        path: Path to image
        
    Returns:
        Dictionary with image info
    """
    path = Path(path)
    
    # Get file info
    stat = path.stat()
    
    # Load image header only
    with Image.open(path) as img:
        width, height = img.size
        mode = img.mode
        format_name = img.format
    
    return {
        "path": str(path),
        "filename": path.name,
        "width": width,
        "height": height,
        "channels": len(mode),
        "mode": mode,
        "format": format_name,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 ** 2),
    }


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Resize image while maintaining aspect ratio with padding.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        padding_color: Color for padding
        
    Returns:
        Tuple of (resized image, padding info dict)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # Apply padding
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=padding_color
        )
    else:
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=padding_color[0]
        )
    
    padding_info = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "original_size": (w, h),
        "resized_size": (new_w, new_h),
    }
    
    return padded, padding_info


def remove_padding(
    image: np.ndarray,
    padding_info: Dict[str, int]
) -> np.ndarray:
    """
    Remove padding from an image using padding info.
    
    Args:
        image: Padded image
        padding_info: Padding information from resize_with_padding
        
    Returns:
        Image with padding removed
    """
    h, w = image.shape[:2]
    
    top = padding_info["pad_top"]
    bottom = h - padding_info["pad_bottom"]
    left = padding_info["pad_left"]
    right = w - padding_info["pad_right"]
    
    return image[top:bottom, left:right]
