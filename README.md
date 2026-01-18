# 🦶 DiaFoot.AI

**Deep Learning for Diabetic Foot Ulcer Segmentation**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art wound segmentation system using U-Net++ with EfficientNet-B4 encoder, achieving **84.93% IoU** and **91.73% Dice** score on the FUSeg dataset.

![DiaFoot.AI Demo](docs/demo.png)

## 🎯 Performance

| Metric | Score | vs SOTA |
|--------|-------|---------|
| **IoU** | 0.8493 | 97% of DFUC 2022 Winner |
| **Dice** | 0.9173 | 99% of target |
| **Inference** | ~50ms | Real-time capable |

## 🏗️ Architecture
```
Input Image (RGB)
      ↓
┌─────────────────────┐
│  CLAHE Enhancement  │  ← Contrast enhancement
└─────────────────────┘
      ↓
┌─────────────────────┐
│    U-Net++          │
│  EfficientNet-B4    │  ← Pretrained encoder
│    Encoder          │
└─────────────────────┘
      ↓
┌─────────────────────┐
│  Post-processing    │  ← Remove noise, fill holes
└─────────────────────┘
      ↓
Wound Segmentation Mask
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
```bash
python scripts/download_datasets.py --all
```

### Training
```bash
# Basic training
python scripts/train_simple.py

# Advanced training (Focal Tversky + EMA)
python scripts/train_advanced.py
```

### Inference
```python
from src.inference.optimized_pipeline import load_pipeline
from PIL import Image
import numpy as np

# Load pipeline
pipeline = load_pipeline("outputs/fuseg_simple/best_model.pt")

# Predict
image = np.array(Image.open("wound_image.jpg").convert("RGB"))
result = pipeline.predict(image)

# Get results
mask = result["mask"]                    # Binary segmentation
wound_pct = result["wound_percentage"]   # Wound coverage %
confidence = result["confidence"]        # Model confidence
```

## 📁 Project Structure
```
DiaFoot.AI/
├── src/
│   ├── models/
│   │   └── segmentation.py      # U-Net++ model
│   ├── data/
│   │   ├── dataset.py           # Data loading
│   │   └── augmentation.py      # Augmentations
│   ├── training/
│   │   └── trainer.py           # Training logic
│   └── inference/
│       ├── enhanced_pipeline.py # Full pipeline
│       └── optimized_pipeline.py# Production pipeline
├── scripts/
│   ├── train_simple.py          # Basic training
│   ├── train_advanced.py        # Advanced training
│   ├── download_datasets.py     # Dataset download
│   └── test_model.py            # Model testing
├── configs/
│   └── config.yaml              # Configuration
├── outputs/                     # Trained models
└── data/                        # Datasets
```

## 🔬 Technical Details

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | U-Net++ |
| Encoder | EfficientNet-B4 (ImageNet pretrained) |
| Input Size | 512 × 512 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 (basic), 3e-4 (advanced) |
| Loss | Dice + BCE / Focal Tversky + BCE |
| Epochs | 100-150 |

### Inference Enhancements

- **CLAHE Preprocessing**: Adaptive histogram equalization for contrast
- **Test Time Augmentation**: Horizontal/vertical flips averaged
- **Post-processing**: Small region removal, hole filling, boundary smoothing

## 📊 Datasets

| Dataset | Images | Used For |
|---------|--------|----------|
| FUSeg 2021 | 1,210 | Training & Validation |
| AZH Wound | 2,849 | Additional training |
| DFUC 2022 | 15,683 | (Requires license) |

## 📈 Results

### Validation Performance
```
Epoch 62: Best Model
├── IoU:  0.8493
├── Dice: 0.9173
├── Train Loss: 0.0486
└── Val Loss: 0.0630
```

### Test Performance (with optimizations)
```
Average IoU:  0.8097
Average Dice: 0.8915
```

## 🛠️ Requirements

- Python 3.11+
- PyTorch 2.0+
- segmentation-models-pytorch
- albumentations
- OpenCV
- NumPy

See `requirements.txt` for full list.

## 📝 Citation

If you use this work, please cite:
```bibtex
@software{diafootai2026,
  author = {Ruthvik Bandari},
  title = {DiaFoot.AI: Deep Learning for Diabetic Foot Ulcer Segmentation},
  year = {2026},
  url = {https://github.com/Ruthvik-Bandari/DiaFoot.AI}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- FUSeg Challenge organizers
- segmentation-models-pytorch library
- Northeastern University AAI6620 Course

---

**Author**: Ruthvik Bandari  
**Course**: AAI6620 Computer Vision, Northeastern University  
**Date**: January 2026
