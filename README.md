# DiaFoot.AI v2 — Diabetic Foot Ulcer Detection & Segmentation

> **⚠️ IMPORTANT DISCLAIMER:** This is an academic research project developed for educational purposes as part of the AAI6620 Computer Vision course at Northeastern University. **This software is NOT a medical device, is NOT FDA-cleared, and is NOT intended for clinical use, diagnosis, treatment, or any medical decision-making.** It does not replace professional medical judgment. Always consult a qualified healthcare provider for any medical concerns. The authors assume no liability for any use of this software in clinical or diagnostic settings.

---

A production-grade multi-task pipeline for automated diabetic foot ulcer (DFU) detection, wound boundary segmentation, and clinical wound assessment. Built to demonstrate modern computer vision techniques applied to medical imaging.

---

## Clinical Motivation

Diabetic foot ulcers affect 15–25% of diabetic patients in their lifetime, with 85% of diabetes-related amputations preceded by a foot ulcer. Early detection and accurate wound measurement can reduce amputation rates by up to 85%. DiaFoot.AI explores how deep learning can automate wound boundary detection to potentially support clinical workflows in the future.

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

**Why cascaded?** The data composition ablation proved that the segmenter performs best when trained exclusively on DFU images (85.89% Dice). Adding non-DFU wounds actually hurt performance (68.71% Dice) because the model gets confused learning two different wound morphologies simultaneously. The classifier handles triage; the segmenter focuses on what it does best.

---

## Results

### 5-Fold Cross-Validated Segmentation (DFU)

The primary result, validated across 5 independent train/val splits for statistical rigor:

| Fold | Dice | IoU |
|------|------|-----|
| 0 | 84.69% | 78.03% |
| 1 | 86.10% | 79.87% |
| 2 | 85.98% | 79.00% |
| 3 | 84.74% | 78.07% |
| 4 | 85.66% | 78.54% |
| **Mean ± Std** | **85.43 ± 0.61%** | **78.70 ± 0.68%** |

The standard deviation of ±0.61% confirms the model performs consistently regardless of data partitioning.

### Test Set Evaluation (DFU, n=285)

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

**Key finding:** Adding 871 more DFU images (AZH wound care center data) improved Dice by +2.17% with no other changes. Data quality and quantity matter more than architecture complexity.

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

### 5-Fold Cross-Validation

```bash
# Submit as SLURM array job (parallel)
sbatch slurm/run_cv.sh

# Or run individual folds
python scripts/run_cross_val.py --fold 0 --device cuda --epochs 100
python scripts/run_cross_val.py --fold 1 --device cuda --epochs 100
# ... folds 2, 3, 4
```

### Evaluation

```bash
python scripts/evaluate.py \
    --task segment \
    --checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --device cuda
```

```bash
python scripts/evaluate.py \
    --task classify \
    --checkpoint checkpoints/classifier/best_epoch004_1.0000.pt \
    --device cuda
# writes: results/classification_metrics.json + results/classification_calibration.json
```

### Leakage Audit (Train/Val/Test)

```bash
python scripts/run_leakage_audit.py \
    --splits-dir data/splits \
    --output data/metadata/leakage_report.json
```

### External Validation Benchmark

```bash
python scripts/run_external_validation.py \
    --internal-split data/splits/test.csv \
    --external-split data/splits/external.csv \
    --cls-checkpoint checkpoints/classifier/best_epoch004_1.0000.pt \
    --seg-checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt
```

### Classifier Shortcut Audit

```bash
python scripts/run_shortcut_audit.py \
    --checkpoint checkpoints/classifier/best_epoch004_1.0000.pt \
    --split-csv data/splits/test.csv \
    --output results/shortcut_audit.json
```

### Subgroup Audit (with CIs)

```bash
python scripts/run_subgroup_audit.py \
    --split-csv data/splits/test.csv \
    --cls-checkpoint checkpoints/classifier/best_epoch004_1.0000.pt \
    --seg-checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --output results/subgroup_audit.json
```

### Failure Atlas (Error Taxonomy)

```bash
python scripts/run_failure_atlas.py \
    --checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --split-csv data/splits/test.csv \
    --output-dir results/failure_atlas
```

### Reproducibility Bundle

```bash
python scripts/run_repro_bundle.py \
    --include pyproject.toml requirements.txt configs/deploy/api.yaml \
    --output results/repro/repro_bundle.json
```

### Clinical Area Agreement Audit

```bash
python scripts/run_area_agreement.py \
    --input-csv results/area_measurements.csv \
    --output results/area_agreement.json
```

### ONNX Export

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --output models/diafoot_segmenter.onnx \
    --validate --benchmark
```

```bash
python scripts/run_onnx_parity.py \
    --checkpoint checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt \
    --onnx models/diafoot_segmenter.onnx \
    --split-csv data/splits/test.csv \
    --output results/onnx_parity_report.json
```

### FastAPI Server

```bash
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
# POST /predict with image file
```

Optional runtime overrides:

```bash
export DIAFOOT_CLASSIFIER_CKPT=checkpoints/classifier/best_epoch004_1.0000.pt
export DIAFOOT_SEGMENTER_CKPT=checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt
export DIAFOOT_CONFIDENCE_THRESHOLD=0.95
export DIAFOOT_DEFER_THRESHOLD=0.67
# or let API auto-read from results/classification_calibration.json
export DIAFOOT_CALIBRATION_PATH=results/classification_calibration.json

# Optional image quality guardrails (manual-review trigger)
export DIAFOOT_MIN_IMAGE_SIDE=256
export DIAFOOT_BLUR_VARIANCE_THRESHOLD=30
export DIAFOOT_BRIGHTNESS_MIN=20
export DIAFOOT_BRIGHTNESS_MAX=235

# Request guardrails
export DIAFOOT_MAX_IMAGE_SIZE_MB=20
export DIAFOOT_RATE_LIMIT_RPM=100

# Optional structured monitoring log (JSONL)
export DIAFOOT_PREDICTION_LOG=logs/api/predictions.jsonl
```

### Deploy without Docker (Render + Vercel)

Backend (Render):

1. Push this repo to GitHub.
2. In Render, create a new **Web Service** from the repo.
3. Use the included [render.yaml](render.yaml) blueprint (recommended), or set manually:
    - Build command: `pip install -r requirements.local.txt`
    - Start command: `uvicorn src.deploy.app:app --host 0.0.0.0 --port $PORT`
4. Set runtime env vars in Render:
    - `DIAFOOT_CLASSIFIER_CKPT=checkpoints/classifier/best_epoch004_1.0000.pt`
    - `DIAFOOT_SEGMENTER_CKPT=checkpoints/ablation_dfu_only/best_epoch090_0.1078.pt`
    - `DIAFOOT_CALIBRATION_PATH=results/classification_calibration.json`
    - `DIAFOOT_DEVICE=cpu`
    - `DIAFOOT_CORS_ORIGINS=https://<your-vercel-domain>.vercel.app`

Frontend (Vercel):

1. Deploy the frontend repo/project on Vercel.
2. Set frontend API URL to the Render backend URL (for example `https://diafoot-api.onrender.com`).
3. Ensure `DIAFOOT_CORS_ORIGINS` on Render includes your Vercel domain.

Local startup tip:

- Run backend commands from the project root, not from a frontend directory.
- Correct root example:

```bash
cd ~/Desktop/"Diafoot CV"
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

The API accepts both env variable styles for checkpoints:

- `DIAFOOT_CLASSIFIER_CKPT` or `DIAFOOT_CLASSIFIER_CHECKPOINT`
- `DIAFOOT_SEGMENTER_CKPT` or `DIAFOOT_SEGMENTER_CHECKPOINT`

---

## Project Structure

```
DiaFoot.AI/
├── configs/                    # YAML configs for training, models, data
├── data/
│   ├── raw/                    # Original datasets (DVC tracked)
│   ├── processed/              # Cleaned, preprocessed 512×512 images
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

Every piece of peer feedback from the AAI6620 course review was mapped to a specific implementation:

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

3. **Limited skin tone diversity.** The dataset is predominantly a single ITA group. Fairness conclusions cannot be generalized to the full Fitzpatrick I–VI spectrum. A clinical system would require validation across diverse skin tones.

4. **Only 2 of 5 architectures were fully trained.** FUSegNet underperformed; MedSAM2 LoRA and nnU-Net v2 were implemented but not trained due to time constraints.

5. **Not validated on standardized benchmarks.** Results are on our own data splits. Comparison against the DFUC 2022 challenge leaderboard would require access to their test set.

---

## Regulatory & Ethical Notice

This project is developed **strictly for academic and educational purposes**. It is part of the AAI6620 Computer Vision coursework at Northeastern University.

**This software:**
- Is **NOT** a medical device as defined by the FDA, EU MDR, or any regulatory body
- Has **NOT** undergone clinical validation, regulatory review, or approval of any kind
- Is **NOT** intended to diagnose, treat, cure, or prevent any disease or medical condition
- Should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment
- Has **NOT** been validated in a prospective clinical setting
- Makes **NO** claims of clinical accuracy, safety, or efficacy

**If you are experiencing a medical emergency or have concerns about a diabetic foot ulcer, contact your healthcare provider immediately.**

Any use of this software for clinical decision-making is strictly prohibited and done entirely at the user's own risk. The authors, Northeastern University, and all affiliated parties disclaim all liability for any harm resulting from the use or misuse of this software.

For information on FDA-cleared wound measurement devices, visit [FDA Medical Device Databases](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm).

---

## v1 → v2 Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.

---

## Citation

If you use this work for academic purposes, please cite:

```
@misc{bandari2026diafoot,
  title={DiaFoot.AI: A Multi-Task Pipeline for Diabetic Foot Ulcer Detection and Segmentation},
  author={Bandari, Ruthvik},
  year={2026},
  institution={Northeastern University},
  course={AAI6620 Computer Vision},
  note={Academic project — not for clinical use}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

This license grants permission for academic and research use. **It does not grant permission for clinical or diagnostic use.**

---

*Built with care for educational impact. Data composition matters more than architecture.*
