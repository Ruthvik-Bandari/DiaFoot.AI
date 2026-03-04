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
└─────────────────────────────────┘
```

## Results

### Data Composition Ablation (Key Experiment)

This is the single most important experiment — proving whether adding negative examples helps clinical performance.

| Training Data | DFU Dice | DFU IoU | DFU NSD@5mm | Non-DFU Dice |
|--------------|----------|---------|-------------|-------------|
| **DFU-only** | **85.1%** | **77.5%** | **95.2%** | 32.5% |
| DFU + non-DFU | 79.0% | 69.0% | 91.2% | 64.1% |
| All (with healthy) | 82.4% | 73.7% | 93.3% | 64.4% |

**Finding:** DFU-only training gives the best raw Dice, but cannot handle non-DFU cases. The multi-class approach trades ~3-6% DFU Dice for the ability to handle all foot types — a worthwhile trade-off for clinical deployment.

### Classification (Triage)

| Metric | Score |
|--------|-------|
| Accuracy | 100.0% |
| F1 (macro) | 1.000 |
| DFU Sensitivity | 1.000 |

**Caveat:** Perfect classification reflects visual dataset differences (different cameras/backgrounds per class) rather than pure clinical feature learning. Cross-site validation with same-source images is needed before clinical deployment.

### Segmentation (Best Model: DFU + non-DFU)

| Metric | DFU Only | Non-DFU |
|--------|----------|---------|
| Dice | 80.4% | 63.8% |
| IoU | 70.7% | 50.7% |
| NSD@5mm | 90.8% | 36.5% |
| HD95 (px) | 25.2 | 79.0 |

### ITA-Stratified Fairness Audit

| Skin Tone (ITA) | Dice | IoU | n |
|----------------|------|-----|---|
| Intermediate | 94.2% | 89.0% | 3 |
| Tan | 87.7% | 78.2% | 4 |
| Very Light | 86.9% | 77.3% | 9 |
| Dark | 82.2% | 72.5% | 96 |
| Brown | 73.3% | 62.6% | 12 |
| Light | 63.3% | 55.4% | 4 |

**Fairness gap: 30.9% Dice** — significant bias concern. Small sample sizes per group (3-12, except Dark=96) limit statistical reliability. Larger, more balanced datasets are needed to validate these findings.

### v1 vs v2 Comparison

| Metric | v1 | v2 (DFU-only) | v2 (Multi-class) |
|--------|-----|--------------|-----------------|
| DFU Dice | 91.7% | 85.1% | 80.4% |
| DFU IoU | 84.9% | 77.5% | 70.7% |
| Specificity | 0% | 0% | 100% |
| Clinical Utility | None | DFU only | Full pipeline |

## Dataset

| Category | Images | Purpose |
|----------|--------|---------|
| DFU (FUSeg + AZH) | 1,010 | Wound segmentation targets |
| Healthy Feet | 3,300 | Negative examples |
| Non-DFU Conditions | 2,686 | Hard negatives |
| **Total** | **6,996** | **Multi-class training** |

All images preprocessed to 512x512 with CLAHE enhancement, doubly-stratified splits by class + ITA skin tone category.

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
| HPC Training | SLURM (H200 GPUs) | Explorer Cluster |

## Quick Start

```bash
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI && git checkout v2-rebuild

pip install uv
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

pytest tests/ -v -m "not slow"
```

## Training

```bash
# Triage classifier (all classes)
python scripts/train.py --task classify --device cuda --epochs 50

# Segmentation (DFU + non-DFU only, excludes healthy)
python scripts/train.py --task segment --device cuda --epochs 100

# Data composition ablation
python scripts/run_ablation.py --variant dfu_only --device cuda --epochs 50
python scripts/run_ablation.py --variant dfu_nondfu --device cuda --epochs 50
python scripts/run_ablation.py --variant all --device cuda --epochs 50
```

## Evaluation

```bash
python scripts/evaluate.py --task segment --checkpoint checkpoints/segmentation_v2/best.pt
python scripts/evaluate.py --task classify --checkpoint checkpoints/classifier/best.pt

# Fairness audit
python scripts/run_fairness_audit.py \
    --cls-checkpoint checkpoints/classifier/best.pt \
    --seg-checkpoint checkpoints/segmentation_v2/best.pt
```

## Inference on New Images

```bash
python scripts/predict.py --image path/to/foot_image.jpg --device cpu
```

## Limitations & Future Work

1. **Classifier accuracy inflation**: 100% accuracy is due to visually distinct datasets, not clinical feature learning. Same-source data needed.
2. **Fairness gap**: 30.9% Dice gap across skin tones. Larger balanced datasets required.
3. **Wagner staging**: Architecture built but not trained (no grade annotations available).
4. **Overfitting**: Train/val loss gap of ~2x. More data and stronger regularization needed.
5. **Single-site evaluation**: No cross-site validation performed.

## Peer Feedback Addressed

| Feedback | From | Implementation |
|----------|------|----------------|
| Skin tone fairness | Ching-Yi Mao, Sudeep K.S. | ITA analysis + stratified audit |
| Uncertainty quantification | Om Patel, Yash Jain | MC Dropout + TTA |
| Boundary metrics (HD95, NSD) | Yucheng Yan | Full metric suite |
| Newer architectures | Kasin W. | MedSAM2 LoRA, FUSegNet, nnU-Net v2 |
| P-scSE attention | Shivam Dubey | FUSegNet architecture |
| Robustness testing | Shivam Dubey | Synthetic degradation suite |
| Calibration analysis | Yash Jain | ECE + temperature scaling |
| Ablation studies | Yucheng Yan, Yash Jain | Data composition ablation |

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

---

**Author**: Ruthvik Bandari
**Course**: AAI6620 Computer Vision, Northeastern University
**Date**: March 2026
