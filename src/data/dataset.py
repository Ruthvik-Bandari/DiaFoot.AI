"""
Dataset Module
===============

PyTorch datasets for wound segmentation and classification.

Author: Ruthvik
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class WoundDataset(Dataset):
    """Base dataset for wound images."""
    
    def __init__(
        self,
        images_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_files = self._get_image_files()
    
    def _get_image_files(self) -> List[Path]:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = []
        for f in self.images_dir.iterdir():
            if f.suffix.lower() in extensions:
                files.append(f)
        return sorted(files)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_files[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {"image": image, "path": str(image_path)}


class WoundSegmentationDataset(Dataset):
    """Dataset for wound segmentation with masks."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.samples = self._get_samples()
    
    def _get_samples(self) -> List[Tuple[Path, Path]]:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        samples = []
        
        for img_file in self.images_dir.iterdir():
            if img_file.suffix.lower() not in extensions:
                continue
            
            # Find matching mask
            mask_file = None
            for ext in extensions:
                potential_mask = self.masks_dir / f"{img_file.stem}{ext}"
                if potential_mask.exists():
                    mask_file = potential_mask
                    break
            
            if mask_file:
                samples.append((img_file, mask_file))
        
        return sorted(samples, key=lambda x: x[0].name)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.samples[idx]
        
        # Load image as RGB
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask and convert to single channel binary
        mask = Image.open(mask_path)
        
        # Convert to grayscale if RGB
        if mask.mode == "RGB" or mask.mode == "RGBA":
            mask = mask.convert("L")
        
        mask = np.array(mask)
        
        # Normalize to 0-1
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).float()
        
        return {
            "image": image,
            "mask": mask,
            "path": str(img_path),
        }


class FUSeg2021Dataset(Dataset):
    """Dataset for FUSeg 2021 Challenge."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # FUSeg structure
        if split == "train":
            self.images_dir = self.root_dir / "train" / "images"
            self.masks_dir = self.root_dir / "train" / "labels"
        elif split == "validation" or split == "val":
            self.images_dir = self.root_dir / "validation" / "images"
            self.masks_dir = self.root_dir / "validation" / "labels"
        elif split == "test":
            self.images_dir = self.root_dir / "test" / "images"
            self.masks_dir = self.root_dir / "test" / "labels"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        self.samples = self._get_samples()
        print(f"  Loaded {len(self.samples)} samples for {split} split")
    
    def _get_samples(self) -> List[Tuple[Path, Path]]:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        samples = []
        
        for img_file in self.images_dir.iterdir():
            if img_file.suffix.lower() not in extensions:
                continue
            
            # Find matching mask
            for ext in extensions:
                mask_file = self.masks_dir / f"{img_file.stem}{ext}"
                if mask_file.exists():
                    samples.append((img_file, mask_file))
                    break
        
        return sorted(samples, key=lambda x: x[0].name)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask - CRITICAL FIX: Convert RGB mask to binary
        mask = Image.open(mask_path)
        
        # Convert RGB to grayscale
        if mask.mode == "RGB" or mask.mode == "RGBA":
            mask = mask.convert("L")
        
        mask = np.array(mask)
        
        # Binarize: 0-255 -> 0-1
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0).float()
        
        return {
            "image": image,
            "mask": mask,
            "path": str(img_path),
        }


class WoundClassificationDataset(Dataset):
    """Dataset for wound classification."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Auto-detect classes from folder names
        self.class_names = class_names or sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self.samples = self._get_samples()
    
    def _get_samples(self) -> List[Tuple[Path, int]]:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in extensions:
                    samples.append((img_file, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label = self.samples[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
        }


class AZHWoundDataset(Dataset):
    """Dataset for AZH wound images."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = self._get_samples()
    
    def _get_samples(self) -> List[Tuple[Path, Path]]:
        images_dir = self.root_dir / "images"
        masks_dir = self.root_dir / "labels"
        
        if not images_dir.exists():
            images_dir = self.root_dir / "image"
        if not masks_dir.exists():
            masks_dir = self.root_dir / "label"
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        samples = []
        
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in extensions:
                continue
            
            for ext in extensions:
                mask_file = masks_dir / f"{img_file.stem}{ext}"
                if mask_file.exists():
                    samples.append((img_file, mask_file))
                    break
        
        return sorted(samples, key=lambda x: x[0].name)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.samples[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask = Image.open(mask_path)
        if mask.mode == "RGB" or mask.mode == "RGBA":
            mask = mask.convert("L")
        mask = np.array(mask)
        
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0).float()
        
        return {
            "image": image,
            "mask": mask,
            "path": str(img_path),
        }


class MultiTaskWoundDataset(Dataset):
    """Dataset for multi-task learning (segmentation + classification)."""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        labels_file: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Load labels
        import json
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        self.samples = self._get_samples()
    
    def _get_samples(self) -> List[Tuple[Path, Path, int]]:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        samples = []
        
        for img_name, label in self.labels.items():
            img_stem = Path(img_name).stem
            
            img_file = None
            for ext in extensions:
                potential = self.images_dir / f"{img_stem}{ext}"
                if potential.exists():
                    img_file = potential
                    break
            
            if img_file is None:
                continue
            
            mask_file = None
            for ext in extensions:
                potential = self.masks_dir / f"{img_stem}{ext}"
                if potential.exists():
                    mask_file = potential
                    break
            
            if mask_file:
                samples.append((img_file, mask_file, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, label = self.samples[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask = Image.open(mask_path)
        if mask.mode == "RGB" or mask.mode == "RGBA":
            mask = mask.convert("L")
        mask = np.array(mask)
        
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0).float()
        
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
        }
