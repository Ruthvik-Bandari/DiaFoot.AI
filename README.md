# DiaFoot.AI 🦶

**Deep Learning for Diabetic Foot Ulcer Segmentation**

[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow.svg)](https://huggingface.co/RuthvikBandari/DiaFootAI)

A state-of-the-art deep learning system for automatic segmentation of diabetic foot ulcers, achieving **85.58% IoU** — surpassing the published MICCAI FUSeg 2021 challenge winner by **+5.28%**.

## Highlights

| Achievement | Value |
|-------------|-------|
| **Best IoU** | 85.58% |
| **Best Dice** | 92.23% |
| **vs Published SOTA** | +5.28% improvement |
| **Cross-Validation** | 84.13% ± 1.30% IoU |
| **Inference Speed** | ~50ms per image |

## Performance Comparison

| Model | IoU | Dice | Source |
|-------|-----|------|--------|
| **DiaFoot.AI (Ours)** | **85.58%** | **92.23%** | This repo |
| x-FUSegNet (FUSeg Winner) | 80.30% | 89.23% | MICCAI 2021 |
| Mahbod et al. | 79.90% | 88.80% | MICCAI 2021 |
| DFUSegNet | 79.06% | 85.76% | 2024 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DiaFoot.AI Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  Input Image (512×512)                                      │
│       ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  U-Net++ with EfficientNet-B4 Encoder               │    │
│  │  - ImageNet pretrained weights                      │    │
│  │  - 20.8M parameters                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                     │
│  Test Time Augmentation (8 transforms)                      │
│       ↓                                                     │
│  Binary Segmentation Mask                                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Techniques

| Technique | Purpose | Impact |
|-----------|---------|--------|
| **U-Net++** | Dense skip connections | Better feature fusion |
| **EfficientNet-B4** | Pretrained encoder | Strong feature extraction |
| **Focal Tversky Loss** | Handle class imbalance | +2% IoU vs BCE |
| **Boundary Loss** | Sharp edge detection | Cleaner boundaries |
| **EMA (0.999)** | Weight smoothing | Stable training |
| **Test Time Augmentation** | 8 transform ensemble | +0.26% IoU |
| **5-Fold Cross-Validation** | Statistical validation | Proven robustness |

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU) or MPS (for Apple Silicon)

### Setup

```bash
# Clone repository
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Model Weights

Download the pretrained model from [Hugging Face](https://huggingface.co/RuthvikBandari/DiaFootAI):

```bash
# Option 1: Using huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('RuthvikBandari/DiaFootAI', 'best_model.pt', local_dir='outputs/fuseg_simple')"

# Option 2: Direct download
wget https://huggingface.co/RuthvikBandari/DiaFootAI/resolve/main/best_model.pt -P outputs/fuseg_simple/
```

Or download directly: [best_model.pt](https://huggingface.co/RuthvikBandari/DiaFootAI/resolve/main/best_model.pt)

## Quick Start

### Inference on Single Image

```python
from src.inference.optimized_pipeline import load_pipeline
from PIL import Image
import numpy as np

# Load model
pipeline = load_pipeline("outputs/fuseg_simple/best_model.pt")

# Predict
image = np.array(Image.open("wound_image.jpg").convert("RGB"))
result = pipeline.predict(image)

# Results
mask = result["mask"]                    # Binary segmentation
wound_pct = result["wound_percentage"]   # Wound area percentage
confidence = result["confidence"]        # Model confidence
```

### Inference with TTA (Recommended)

```bash
python src/inference/tta_inference.py \
    --checkpoint outputs/fuseg_simple/best_model.pt \
    --encoder efficientnet-b4 \
    --image path/to/image.jpg \
    --output path/to/output.png
```

### Batch Evaluation

```bash
python src/inference/tta_inference.py \
    --checkpoint outputs/fuseg_simple/best_model.pt \
    --encoder efficientnet-b4 \
    --data-dir "data/raw/fuseg/wound-segmentation/data/Foot Ulcer Segmentation Challenge"
```

## Training

### Basic Training

```bash
python scripts/train_simple.py \
    --encoder efficientnet-b4 \
    --epochs 100 \
    --batch-size 8 \
    --output-dir outputs/my_model
```

### Advanced Training (Recommended)

```bash
python scripts/train_advanced.py \
    --encoder efficientnet-b4 \
    --epochs 100 \
    --batch-size 8 \
    --boundary-loss \
    --output-dir outputs/my_model
```

### 5-Fold Cross-Validation

```bash
python scripts/train_crossval.py \
    --encoder efficientnet-b4 \
    --epochs 80 \
    --output-dir outputs/crossval
```

## Project Structure

```
DiaFoot.AI/
├── src/
│   ├── models/
│   │   └── segmentation.py          # U-Net++ model
│   ├── data/
│   │   ├── dataset.py               # Data loading
│   │   └── augmentation.py          # Augmentation pipeline
│   ├── training/
│   │   └── trainer.py               # Training logic
│   └── inference/
│       ├── tta_inference.py         # TTA inference
│       ├── ensemble_inference.py    # Ensemble inference
│       └── optimized_pipeline.py    # Production pipeline
├── scripts/
│   ├── train_simple.py              # Basic training
│   ├── train_advanced.py            # Advanced training
│   ├── train_crossval.py            # Cross-validation
│   ├── visualize.py                 # Visualization tools
│   └── export_model.py              # ONNX/TorchScript export
├── configs/
│   └── config.yaml                  # Configuration
├── outputs/
│   ├── fuseg_simple/                # Trained models
│   ├── crossval/                    # CV fold models
│   ├── exported/                    # ONNX/TorchScript
│   └── visualizations/              # Sample outputs
└── data/
    └── raw/                         # Datasets
```

## Results

### Training Curves

| Metric | Final Value |
|--------|-------------|
| Training Loss | 0.0486 |
| Validation Loss | 0.0630 |
| Best Epoch | 62 |

### Cross-Validation Results

| Fold | IoU | Dice |
|------|-----|------|
| Fold 1 | 83.22% | 90.84% |
| Fold 2 | 85.81% | 92.36% |
| Fold 3 | 82.16% | 90.20% |
| Fold 4 | 85.00% | 91.89% |
| Fold 5 | 84.44% | 91.56% |
| **Mean** | **84.13% ± 1.30%** | **91.37% ± 0.77%** |

### Test Time Augmentation Results

| Configuration | IoU | Dice |
|---------------|-----|------|
| Without TTA | 85.32% | 92.08% |
| With TTA | **85.58%** | **92.23%** |
| Improvement | +0.26% | +0.15% |

## Model Export

### ONNX (for deployment)

```bash
python scripts/export_model.py \
    --checkpoint outputs/fuseg_simple/best_model.pt \
    --encoder efficientnet-b4 \
    --formats onnx \
    --output-dir outputs/exported
```

### TorchScript (for production PyTorch)

```bash
python scripts/export_model.py \
    --checkpoint outputs/fuseg_simple/best_model.pt \
    --encoder efficientnet-b4 \
    --formats torchscript \
    --output-dir outputs/exported
```

## Dataset

This project uses the [FUSeg 2021 Challenge Dataset](https://github.com/uwm-bigdata/wound-segmentation):

| Split | Images | Usage |
|-------|--------|-------|
| Training | 810 | Model training |
| Validation | 200 | Evaluation |
| Test | 200 | Challenge submission |

## Requirements

See [requirements.txt](requirements.txt) for full list. Key dependencies:

- torch >= 2.0.0
- segmentation-models-pytorch >= 0.3.3
- albumentations >= 1.4.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0

## Citation

If you use this work in your research, please cite:

```bibtex
@software{bandari2025diafootai,
  author = {Bandari, Ruthvik},
  title = {DiaFoot.AI: Deep Learning for Diabetic Foot Ulcer Segmentation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Ruthvik-Bandari/DiaFoot.AI},
  note = {Achieves 85.58\% IoU, surpassing MICCAI FUSeg 2021 winner by +5.28\%}
}
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes without permission

See [LICENSE](LICENSE) for details.

For commercial licensing, contact: bandari.ru@northeastern.edu

## Acknowledgments

- [FUSeg Challenge](https://github.com/uwm-bigdata/wound-segmentation) organizers for the dataset
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) library
- Northeastern University College of Professional Studies

## Author

**Ruthvik Bandari**  
Master's in Applied Artificial Intelligence  
Northeastern University, College of Professional Studies

- GitHub: [@Ruthvik-Bandari](https://github.com/Ruthvik-Bandari)
- LinkedIn: [Ruthvik Bandari](https://linkedin.com/in/ruthvik-bandari)
- Email: bandari.ru@northeastern.edu

---

**Disclaimer:** DiaFoot.AI is intended as a research tool and decision support system. It is not a replacement for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.
