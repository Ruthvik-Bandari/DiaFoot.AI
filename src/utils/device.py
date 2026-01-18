"""
Device and Reproducibility Utilities
=====================================

Functions for device management and ensuring reproducibility.
"""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Priority: specified device > CUDA > MPS (Apple Silicon) > CPU
    
    Args:
        device: Specific device to use ('cuda', 'mps', 'cpu', or None for auto)
        
    Returns:
        torch.device object
    """
    if device is not None:
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cpu": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    return info


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(num_params: int) -> str:
    """
    Format parameter count to human-readable string.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string (e.g., "25.6M")
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 512, 512)) -> dict:
    """
    Get a summary of the model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        
    Returns:
        Dictionary with model summary
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_params_formatted": format_parameters(total_params),
        "trainable_params_formatted": format_parameters(trainable_params),
        "model_size_mb": round(model_size_mb, 2),
    }


def move_to_device(
    data: Union[torch.Tensor, dict, list, tuple],
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively move data to device.
    
    Args:
        data: Tensor, dict, list, or tuple of tensors
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


class DeviceContext:
    """
    Context manager for temporarily changing device.
    
    Example:
        >>> with DeviceContext('cpu'):
        ...     # Operations run on CPU
        ...     pass
    """
    
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.previous_device = None
    
    def __enter__(self):
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def empty_cache():
    """Clear GPU/MPS memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing
        pass


def get_memory_usage() -> dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory information
    """
    info = {}
    
    if torch.cuda.is_available():
        info["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        info["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
        info["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return info


def get_device(device_name: str = "auto"):
    """Get the best available device."""
    import torch
    
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_name)


def print_device_info(device):
    """Print device information."""
    import torch
    
    print(f"\n{'='*50}")
    print(f"Device Information")
    print(f"{'='*50}")
    print(f"  Device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print(f"  GPU: Apple Silicon (MPS)")
        print(f"  Memory: Shared with system RAM")
    else:
        print(f"  CPU Mode")
    print(f"{'='*50}\n")
