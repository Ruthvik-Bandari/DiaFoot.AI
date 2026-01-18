# DiaFootAI — Diabetic Foot Wound Assessment System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Development](https://img.shields.io/badge/Status-Development-orange.svg)]()

## Mission

Enable any healthcare worker with a smartphone to accurately assess diabetic foot wounds, detect infections early, and prevent amputations in underserved communities.

## Overview

DiaFootAI is an AI-powered system that analyzes smartphone images of diabetic foot wounds to provide:

- **Wound Segmentation**: Precise boundary detection for accurate area measurement
- **Tissue Classification**: Identify granulation, slough, necrotic, and epithelial tissue
- **Infection Detection**: Early warning signs of infection requiring urgent referral
- **Healing Tracking**: Monitor wound progression over time
- **Risk Stratification**: Prioritize cases needing immediate attention

## Key Features

| Feature | Description |
|---------|-------------|
| Offline-First | Works without internet connectivity |
| Low-Resource Optimized | Runs on mid-range Android devices |
| Dark Skin Tone Support | Trained on diverse skin tones for equitable performance |
| Clinical Decision Support | Evidence-based recommendations |
| Multi-Language | Support for Hindi, English, and regional languages |

## Project Structure

```
DiaFootAI/
├── configs/                 # Configuration files
│   └── config.yaml         # Main configuration
├── data/
│   ├── raw/                # Original downloaded datasets
│   ├── processed/          # Preprocessed data ready for training
│   ├── annotations/        # Annotation files (JSON, masks)
│   └── external/           # External datasets
├── models/
│   ├── checkpoints/        # Training checkpoints
│   ├── exported/           # Production-ready models (TFLite, CoreML)
│   └── configs/            # Model architecture configs
├── src/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training loops and utilities
│   ├── inference/          # Inference pipelines
│   ├── evaluation/         # Metrics and evaluation
│   └── utils/              # Helper functions
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Standalone scripts
├── tests/                  # Unit and integration tests
├── app/
│   ├── android/            # Android application
│   └── ios/                # iOS application
└── docs/                   # Documentation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- macOS with Apple Silicon (M1/M2/M3/M4) for MLX support
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/DiaFootAI.git
cd DiaFootAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mlx; print('MLX installed successfully')"
```

## Quick Start

### 1. Download Dataset

```bash
python scripts/download_dataset.py --dataset dfuc2022
```

### 2. Preprocess Data

```bash
python scripts/preprocess_data.py --config configs/config.yaml
```

### 3. Train Model

```bash
python scripts/train.py --config configs/config.yaml --model segmentation
```

### 4. Run Inference

```bash
python scripts/inference.py --image path/to/wound_image.jpg
```

## Models

| Model | Task | Architecture | Size | Accuracy |
|-------|------|--------------|------|----------|
| Quality Checker | Image quality assessment | MobileNetV3-Small | 6 MB | - |
| Segmenter | Wound boundary detection | U-Net++ (EfficientNet-B4) | 25 MB | IoU: TBD |
| Tissue Classifier | Tissue type classification | EfficientNet-B3 + ViT | 35 MB | Acc: TBD |
| Infection Detector | Infection risk scoring | ResNet-50 + Attention | 20 MB | Sens: TBD |

## Datasets

| Dataset | Size | Description | Status |
|---------|------|-------------|--------|
| DFUC 2022 | 15,683 | Diabetic foot ulcer classification | Planned |
| FUSeg 2021 | 1,210 | Foot ulcer segmentation | Planned |
| Medetec | 500+ | Various chronic wounds | Planned |
| Custom (India) | 500+ | Indian population data | Collection TBD |

## Development Roadmap

- [ ] **Month 1**: Dataset preparation, baseline models
- [ ] **Month 2**: Core segmentation model
- [ ] **Month 3**: Tissue classification, infection detection
- [ ] **Month 4**: Mobile app development
- [ ] **Month 5**: Integration and testing
- [ ] **Month 6**: Clinical pilot
- [ ] **Month 7**: Iteration and refinement
- [ ] **Month 8**: Open source release

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{diafootai2025,
  title = {DiaFootAI: Diabetic Foot Wound Assessment System},
  author = {Ruthvik},
  year = {2025},
  url = {https://github.com/yourusername/DiaFootAI}
}
```

## Acknowledgments

- DFUC Challenge organizers for the dataset
- Medical partners for clinical guidance
- Open source community for foundational tools

## Contact

- **Developer**: Ruthvik
- **Email**: [your-email]
- **LinkedIn**: [your-linkedin]

---

**Disclaimer**: DiaFootAI is intended as a screening aid and decision support tool. It is not a replacement for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.
