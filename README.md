# DiaFoot.AI v2 — Diabetic Foot Ulcer Detection, Classification & Segmentation

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade multi-task deep learning system for automated diabetic foot ulcer (DFU) detection. Unlike v1 (binary segmentation only), v2 implements a three-stage clinical pipeline: **triage classification**, **wound segmentation**, and **severity staging**.

## Why v2?

v1 achieved 84.93% IoU but had zero clinical specificity — it was trained only on ulcer images, so it predicted wounds on every input (including healthy feet). v2 fixes this fundamental flaw by training on three categories: healthy feet, non-DFU conditions, and DFU across all Wagner grades.

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│  TASK 1: Triage Classification  │
│  EfficientNet-V2-M backbone     │
│  Output: Healthy | Non-DFU | DFU│
└─────────────┬───────────────────┘
              │ (Only if DFU detected)
              ▼
┌─────────────────────────────────┐
│  TASK 2: DFU Segmentation       │
│  U-Net++ / EfficientNet-B4      │
│  Output: Pixel-wise wound mask  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  TASK 3: Severity Staging       │
│  Wagner Grade 0-5 classifier    │
└─────────────────────────────────┘
```

## Results

### Classification (Triage)

| Metric | Score |
|--------|-------|
| Accuracy | 100.0% |
| F1 (macro) | 1.000 |
| DFU Sensitivity | 1.000 |
| AUROC | 1.000 |

Note: Perfect classification likely reflects visual dataset differences rather than pure clinical feature learning. See Discussion section in the report.

### Segmentation

| Metric | Overall | DFU Only | Non-DFU |
|--------|---------|----------|---------|
| Dice | 83.4% | 81.2% | 64.7% |
| IoU | 76.9% | 71.9% | 51.4% |
| NSD@5mm | 74.9% | 92.0% | 37.5% |
| HD95 (px) | 31.5 | 20.9 | 74.3 |

### v1 vs v2 Comparison

| Metric | v1 | v2 (DFU) | Improvement |
|--------|-----|----------|-------------|
| IoU | 84.9% | 71.9% | Multi-class (harder task) |
| Dice | 91.7% | 81.2% | Multi-class (harder task) |
| Specificity | 0% | 100% | Can identify healthy feet |
| Clinical Utility | None | High | Three-stage pipeline |

## Dataset

| Category | Images | Purpose |
|----------|--------|---------|
| DFU (FUSeg + AZH) | 1,010 | Wound segmentation targets |
| Healthy Feet | 3,300 | Negative examples (no wound) |
| Non-DFU Conditions | 2,686 | Hard negatives (wounds that are not DFU) |
| **Total** | **6,996** | **Multi-class training** |

All images preprocessed to 512x512 with CLAHE enhancement, doubly-stratified splits (by class + ITA skin tone category).

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Deep Learning | PyTorch | 2.10.0 |
| Medical Imaging | MONAI | 1.5.2 |
| 2D Segmentation | SMP | 0.5.0 |
| Augmentation | Albumentations | 1.4.x (MIT) |
| Data Quality | CleanVision + Cleanlab | Latest |
| API | FastAPI | 0.133.0 |
| Linting | Ruff | 0.15.2 |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI
git checkout v2-rebuild

# Environment
pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v -m "not slow"
```

## Training

```bash
# Triage classifier
python scripts/train.py --config configs/training/baseline.yaml --task classify --device cuda --epochs 50

# Segmentation
python scripts/train.py --config configs/training/baseline.yaml --task segment --device cuda --epochs 50
```

## Evaluation

```bash
python scripts/evaluate.py --task classify --checkpoint checkpoints/classifier/best.pt
python scripts/evaluate.py --task segment --checkpoint checkpoints/segmentation/best.pt
```

## Project Structure

```
DiaFoot.AI-v2/
├── src/
│   ├── data/           # Preprocessing, augmentation, ITA analysis
│   ├── models/         # Classifier, U-Net++, FUSegNet, MedSAM2 LoRA
│   ├── training/       # Losses, trainer, multi-task trainer, EMA
│   ├── evaluation/     # Metrics, fairness, calibration, uncertainty
│   ├── inference/      # TTA, ONNX export, pipeline
│   └── deploy/         # FastAPI REST service
├── configs/            # YAML configurations
├── scripts/            # Training and evaluation entry points
├── slurm/              # HPC job scripts
├── tests/              # Comprehensive test suite
└── data/
    ├── raw/            # Original datasets
    ├── processed/      # Preprocessed 512x512 images
    ├── splits/         # Train/val/test CSVs
    └── metadata/       # Quality reports, ITA scores, Wagner grades
```

## Peer Feedback Addressed

| Feedback | From | Implementation |
|----------|------|----------------|
| Skin tone fairness analysis | Ching-Yi Mao, Sudeep K.S. | ITA-stratified metrics (Commit 6, 26) |
| Uncertainty quantification | Om Patel, Yash Jain | MC Dropout + TTA (Commits 23-24) |
| Inter-annotator comparison | Om Patel, Nada Moursi | STAPLE consensus (Commit 21) |
| Boundary metrics (HD95, NSD) | Yucheng Yan | Full metric suite (Commit 20) |
| Newer architectures | Kasin W. | MedSAM2 LoRA, nnU-Net v2 (Commits 10-11) |
| P-scSE attention | Shivam Dubey | FUSegNet architecture (Commit 12) |
| Robustness testing | Shivam Dubey | Synthetic degradation suite (Commit 22) |
| Calibration analysis | Yash Jain | ECE + temperature scaling (Commit 25) |

## Citation

```bibtex
@software{diafootai_v2_2026,
  author = {Ruthvik Bandari},
  title = {DiaFoot.AI v2: Multi-Task Deep Learning for Diabetic Foot Ulcer Detection},
  year = {2026},
  url = {https://github.com/Ruthvik-Bandari/DiaFoot.AI}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- FUSeg Challenge organizers
- Segmentation Models PyTorch library
- Northeastern University AAI6620 Computer Vision Course
- Peer reviewers who provided invaluable feedback

---

**Author**: Ruthvik Bandari
**Course**: AAI6620 Computer Vision, Northeastern University
**Date**: March 2026
