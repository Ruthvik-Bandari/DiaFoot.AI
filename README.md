# DiaFoot.AI v2 — Diabetic Foot Ulcer Detection & Segmentation

A production-grade multi-task pipeline for automated diabetic foot ulcer (DFU) detection, wound boundary segmentation, and clinical wound assessment. Built for AAI6620 Computer Vision at Northeastern University.

---

## Clinical Motivation

Diabetic foot ulcers affect 15–25% of diabetic patients in their lifetime, with 85% of diabetes-related amputations preceded by a foot ulcer. Early detection and accurate wound measurement can reduce amputation rates by up to 85%. DiaFoot.AI automates wound boundary detection to support clinical decision-making.

## Why v2: Lessons from v1

The original DiaFoot.AI (v1) achieved 84.93% IoU and 91.73% Dice — numbers that looked strong but masked two fundamental flaws:

1. **Training data contained only ulcer images.** The model never learned what healthy skin looks like, so it predicted ulcers on every input — zero clinical specificity. A model that calls everything a wound is clinically useless.
2. **No data cleaning pipeline.** Raw scraped images were fed directly into training with no quality audit, duplicate detection, or label verification.

DiaFoot.AI v2 is a ground-up rebuild that fixes both problems through a multi-task cascaded pipeline and rigorous data engineering.

---

## Architecture

The system uses a **cascaded pipeline** (Strategy A), validated by ablation to outperform joint multi-task training:

```
Input Image
    │
    ▼
┌───────────────────────────┐
│  Triage Classifier        │
│  EfficientNet-V2-M        │
│                           │
│  → Healthy                │  ← Stop. No wound detected.
│  → Non-DFU Condition      │  ← Stop. Not a diabetic ulcer.
│  → DFU Detected           │  ← Proceed to segmentation.
└───────────┬───────────────┘
            │ (DFU only)
            ▼
┌───────────────────────────┐
│  Wound Segmenter          │
│  U-Net++ / EfficientNet-B4│
│  + scSE Attention         │
│                           │
│  → Pixel-wise wound mask  │
│  → Wound area (mm²)       │
│  → Boundary metrics       │
└───────────────────────────┘
```

**Why cascaded?** The data composition ablation proved that the segmenter performs best when trained exclusively on DFU images (85.89% Dice). Adding non-DFU wounds actually *hurt* performance (68.71% Dice) because the model gets confused learning two different wound morphologies simultaneously. The classifier handles triage; the segmenter focuses on what it does best.

---

## Results

### Segmentation Performance (DFU Test Set, n=285)

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Dice Score** | **85.89%** | Strong pixel-level wound overlap |
| **IoU (Jaccard)** | **79.35%** | Solid intersection accuracy |
| **HD95** | **17.3 px** | 95th percentile boundary distance |
| **NSD@2mm** | **85.86%** | 86% of predicted boundary within 2mm of ground truth |
| **NSD@5mm** | **94.74%** | 95% within 5mm — clinically excellent |
| **Wound Area Error** | **1.1%** | Predicted 1,342 mm² vs ground truth 1,358 mm² |

### Data Composition Ablation

The single most important experiment — proving that data composition matters more than architecture:

| Training Data | Best Dice | Val Loss | Overfitting Ratio |
|---------------|-----------|----------|-------------------|
| **DFU-only (1,881 images)** | **87.44%** | 0.1078 | 1.3x (minimal) |
| DFU-only (1,010 images) | 85.27% | 0.1057 | 1.0x (none) |
| DFU + non-DFU | 68.71% | 0.4187 | 1.4x |
| All classes (DFU + healthy + non-DFU) | 84.14%* | 0.6723 | 2.9x (heavy) |

*\*Inflated by healthy images scoring perfectly on empty masks.*

**Key finding:** Adding 871 more DFU images (AZH wound care center data) improved Dice by +2.17% with no other changes. Data quality > architecture complexity.

### Architecture Comparison

| Model | Best Dice | Parameters | Notes |
|-------|-----------|------------|-------|
| **U-Net++ / EfficientNet-B4 + scSE** | **87.44%** | ~25M | Best performance |
| FUSegNet / EfficientNet-B7 + P-scSE | 69.60% | ~66M | Too many parameters for dataset size |

### Test-Time Augmentation (TTA)

| Metric | Without TTA | With TTA (16-aug) | Improvement |
|--------|-------------|-------------------|-------------|
| Dice | 57.38%* | 61.26%* | +3.88% |
| IoU | 52.28%* | 56.29%* | +4.01% |
| HD95 | 87.47 | 84.88 | -2.59 (better) |

*\*Overall numbers including non-DFU images; DFU-specific TTA improvement follows the same trend.*

### Fairness Analysis (ITA-Stratified)

| ITA Group | Count | Dice | IoU | HD95 |
|-----------|-------|------|-----|------|
| Brown | 285 | 85.89% | 79.35% | 17.3 |
| **Fairness gap** | — | **0.00%** | — | — |

**Limitation:** The dataset is predominantly composed of a single ITA skin tone group (Brown). While no fairness gap exists within the represented population, the model has not been validated across the full Fitzpatrick I–VI spectrum. ITA computation on wound images is confounded by wound bed color; a clinical deployment would require ITA measurement from non-wound skin regions specifically.

---

## Dataset

### Composition (8,105 total samples)

| Category | Images | Purpose |
|----------|--------|---------|
| **DFU** | 2,119 | Wound segmentation training (FUSeg + AZH) |
| **Healthy Feet** | 3,300 | True negatives for classifier |
| **Non-DFU Conditions** | 2,686 | Hard negatives (general wounds, not DFU) |

### Sources

| Dataset | Images | Type |
|---------|--------|------|
| FUSeg 2021 (UWM BigData Lab) | 1,010 | DFU with segmentation masks |
| AZH Wound Care Center | 1,109 | Clinical wound patches with masks |
| Kaggle DFU Patches | 543 | Healthy foot skin patches |
| Mendeley Wound Dataset (Normal) | 2,757 | Healthy foot images |
| Mendeley Wound Dataset (Wounds) | 2,686 | Non-DFU wound images with masks |

### Data Pipeline

All images pass through a production cleaning pipeline:

1. **Integrity check** — verify every image opens and is not corrupt
2. **Mask validation** — binary format check, dimension alignment, coverage statistics
3. **Deduplication** — perceptual hash (dHash) to remove cross-dataset duplicates
4. **Preprocessing** — resize to 512×512 (aspect-preserving pad), CLAHE contrast enhancement, mask binarization
5. **Stratified splits** — 70/15/15 train/val/test, stratified by class and ITA skin tone group, zero data leakage verified

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Deep Learning | PyTorch | 2.10.0 |
| Medical Imaging | MONAI | 1.5.2 |
| Segmentation | Segmentation Models PyTorch | 0.5.0 |
| Augmentation | Albumentations | 1.4.24 |
| Data Quality | CleanVision, Cleanlab | Latest |
| API | FastAPI | 0.133.0 |
| Inference | ONNX Runtime | 1.24.2 |
| Linting | Ruff | 0.15.2 |
| Compute | Northeastern Explorer HPC (H200/A100 GPUs) | — |

---

## Quick Start

### Installation

```bash
git clone https://github.com/Ruthvik-Bandari/DiaFoot.AI.git
cd DiaFoot.AI
pip install -r requirements.txt
```

### Inference (Single Image)

```bash
python scripts/predict.py --image path/to/foot.jpg --device cuda
```

### Training

```bash
# Train classifier (all 3 classes)
python scripts/train.py --task classify --epochs 50 --device cuda

# Train segmenter (DFU-only — proven best by ablation)
python scripts/run_ablation.py --variant dfu_only --epochs 100 --device cuda
```

### Evaluation

```bash
python scripts/evaluate.py \
    --task segment \
    --checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --device cuda
```

### ONNX Export

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --output models/diafoot_segmenter.onnx \
    --validate --benchmark
```

### FastAPI Server

```bash
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
# POST /predict with image file
```

---

## Project Structure

```
DiaFoot.AI/
├── configs/                    # YAML configs for training, models, data
├── data/
│   ├── raw/                    # Original datasets (DVC tracked)
│   ├── processed/              # Cleaned, preprocessed 512x512 images
│   ├── splits/                 # Train/val/test CSVs
│   └── metadata/               # Quality reports, ITA scores
├── src/
│   ├── data/                   # Dataset classes, augmentation, cleaning
│   ├── models/                 # U-Net++, FUSegNet, classifier, MedSAM2
│   ├── training/               # Trainer, losses, schedulers, EMA
│   ├── evaluation/             # Metrics, fairness, calibration, robustness
│   ├── inference/              # Pipeline, TTA, postprocessing
│   └── deploy/                 # FastAPI app
├── scripts/                    # Entry points for train, eval, export
├── slurm/                      # HPC job scripts
├── results/                    # Metrics, figures, reports
├── checkpoints/                # Trained model weights
└── tests/                      # Unit tests
```

---

## Peer Feedback Integration

Every piece of peer feedback from the AAI6620 course review was mapped to a specific implementation. Key examples:

| Feedback | From | Implementation |
|----------|------|----------------|
| How did augmentation handle skin tone diversity? | Sudeep K.S. | ITA-stratified fairness audit |
| Add attention mechanisms to reduce false positives | Shivam Dubey | scSE attention in U-Net++ decoder |
| Report performance relative to inter-annotator agreement | Yucheng Yan | Ceiling analysis framework |
| Tie uncertainty to clinical output | Yash Jain | TTA-based uncertainty maps |
| Prioritize ablation studies over deployment | Yucheng Yan | Data composition ablation as core experiment |
| Addressing algorithmic bias is a critical ethical hurdle | Ching-Yi Mao | ITA fairness audit with honest limitation disclosure |

---

## Honest Limitations

1. **Classifier accuracy (100%) is a dataset artifact.** The three data categories come from visually distinct sources (different cameras, backgrounds). The classifier learned "which dataset" rather than "which condition." A production system requires same-source data across all classes.

2. **Wagner staging was not trained.** The architecture supports it, but clinical grade labels were unavailable. This is acknowledged as future work requiring clinical partnerships.

3. **No cross-validation.** Results are from a single train/val/test split. 5-fold cross-validation would strengthen statistical claims but was not performed due to compute time constraints.

4. **Limited skin tone diversity.** The dataset is predominantly a single ITA group. Fairness conclusions cannot be generalized to the full Fitzpatrick I–VI spectrum.

5. **Only 2 of 5 architectures were fully trained.** FUSegNet underperformed; MedSAM2 LoRA and nnU-Net v2 were implemented but not trained due to time constraints.

---

## v1 → v2 Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.

---

## Citation

If you use this work, please cite:

```
@misc{bandari2026diafoot,
  title={DiaFoot.AI: Production-Grade Diabetic Foot Ulcer Detection and Segmentation},
  author={Bandari, Ruthvik},
  year={2026},
  institution={Northeastern University},
  course={AAI6620 Computer Vision}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with care for clinical impact. Data composition matters more than architecture.*
