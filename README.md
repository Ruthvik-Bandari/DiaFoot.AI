# DiaFoot.AI v2 — Diabetic Foot Ulcer Detection & Segmentation

> **IMPORTANT DISCLAIMER:** This is an academic research project developed for educational purposes as part of the AAI6620 Computer Vision course at Northeastern University. **This software is NOT a medical device, is NOT FDA-cleared, and is NOT intended for clinical use, diagnosis, treatment, or any medical decision-making.** It does not replace professional medical judgment. Always consult a qualified healthcare provider for any medical concerns. The authors assume no liability for any use of this software in clinical or diagnostic settings.

---

A production-grade multi-task pipeline for automated diabetic foot ulcer (DFU) detection, wound boundary segmentation, and clinical wound assessment. Built with **Meta's DINOv2** self-supervised Vision Transformer for robust transfer learning.

---

## Clinical Motivation

Diabetic foot ulcers affect 15-25% of diabetic patients in their lifetime, with 85% of diabetes-related amputations preceded by a foot ulcer. Early detection and accurate wound measurement can reduce amputation rates by up to 85%. DiaFoot.AI explores how deep learning can automate wound boundary detection to potentially support clinical workflows in the future.

## Why v2: Lessons from v1

The original DiaFoot.AI (v1) achieved 84.93% IoU and 91.73% Dice -- numbers that looked strong but masked two fundamental flaws:

1. **Training data contained only ulcer images.** The model never learned what healthy skin looks like, so it predicted ulcers on every input -- zero clinical specificity.
2. **No data cleaning pipeline.** Raw scraped images were fed directly into training with no quality audit, duplicate detection, or label verification.

DiaFoot.AI v2 is a ground-up rebuild that fixes both problems through a multi-task cascaded pipeline, rigorous data engineering, and DINOv2 transfer learning.

---

## Architecture

The system uses a **cascaded pipeline** validated by ablation to outperform joint multi-task training:

```
Input Image (518x518)
    |
    v
+---------------------------+
|  Triage Classifier        |
|  DINOv2 ViT-B/14          |
|  (frozen backbone + head) |
|                           |
|  -> Healthy               |  <- Stop. No wound detected.
|  -> Non-DFU Condition     |  <- Defer to clinician.
|  -> DFU Detected          |  <- Proceed to segmentation.
+----------+----------------+
           | (DFU only)
           v
+---------------------------+
|  Wound Segmenter          |
|  DINOv2 ViT-B/14          |
|  + UPerNet Decoder        |
|                           |
|  -> Pixel-wise wound mask |
|  -> Wound area (mm2)      |
|  -> Boundary metrics      |
+---------------------------+
```

### Why DINOv2?

DINOv2 is Meta's self-supervised Vision Transformer trained on 142M curated images. Key advantages:

- **Domain-agnostic features** -- DINOv2 learns semantic structure (shape, texture, boundaries) rather than camera/dataset-specific shortcuts
- **Parameter efficient** -- Only 0.2% of parameters are trainable (classifier head), yet achieves 98.36% accuracy
- **Strong transfer** -- Achieves 82.73% Dice on wound segmentation with just 10 epochs on a frozen backbone
- **Better calibration** -- ECE of 0.0075 after temperature scaling enables reliable clinical deferral

### Why Cascaded?

The data composition ablation proved that the segmenter performs best when trained exclusively on DFU images (87.44% Dice with U-Net++). Adding non-DFU wounds actually hurt performance (68.71% Dice). The classifier handles triage; the segmenter focuses on DFU morphology.

---

## Results

### DINOv2 Classification (Test Set, n=1,161)

| Metric | Value |
|--------|-------|
| **Accuracy** | **98.36%** |
| **F1 (macro)** | **98.12%** |
| **DFU Sensitivity** | **96.58%** |
| **AUROC** | **99.91%** |
| **ECE (calibrated)** | **0.0075** |
| **Defer Coverage** | 93.45% kept at 99.72% accuracy |

Per-class breakdown:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Healthy | 0.99 | 1.00 | 1.00 |
| Non-DFU | 0.98 | 0.98 | 0.98 |
| DFU | 0.98 | 0.97 | 0.97 |

### DINOv2 Segmentation (Test Set, DFU only, n=263)

| Metric | DINOv2 (10 epochs, frozen) | Previous U-Net++ (90 epochs) |
|--------|---------------------------|------------------------------|
| **Dice Score** | **82.73%** | 85.89% |
| **IoU (Jaccard)** | **74.18%** | 79.35% |
| **HD95** | **19.24 px** | 17.3 px |
| **NSD@2mm** | **75.66%** | 85.86% |
| **NSD@5mm** | **92.84%** | 94.74% |
| **Wound Area** | 1,332 mm2 predicted vs 1,299 mm2 GT | +/- 1.1% |

### Data Composition Ablation

The single most important experiment -- proving that data composition matters more than architecture:

| Training Data | Best Dice | Val Loss |
|---------------|-----------|----------|
| **DFU-only (1,881 images)** | **87.44%** | 0.1078 |
| DFU-only (1,010 images) | 85.27% | 0.1057 |
| DFU + non-DFU | 68.71% | 0.4187 |
| All classes | 84.14%* | 0.6723 |

*Inflated by healthy images scoring perfectly on empty masks.*

### Training Efficiency

| Metric | DINOv2 | Previous Models |
|--------|--------|----------------|
| Trainable params (classifier) | 199K (0.2%) | 54M (100%) |
| Trainable params (segmenter) | 11.6M (11.8%) | 25M (100%) |
| Classifier training time | 6 min (10 epochs) | 30+ min (50 epochs) |
| Segmenter training time | 2 min (10 epochs) | 90+ min (90 epochs) |

### Fairness Analysis (ITA-Stratified)

| ITA Group | Count | Dice | IoU | HD95 |
|-----------|-------|------|-----|------|
| Brown | 263 | 82.73% | 74.18% | 19.24 |
| **Fairness gap** | -- | **0.00%** | -- | -- |

**Limitation:** The dataset is predominantly composed of a single ITA skin tone group. The model has not been validated across the full Fitzpatrick I-VI spectrum.

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

1. **Integrity check** -- verify every image opens and is not corrupt
2. **Mask validation** -- binary format check, dimension alignment, coverage statistics
3. **Deduplication** -- perceptual hash (dHash) to remove cross-dataset duplicates
4. **Preprocessing** -- resize to 512x512 (aspect-preserving pad), CLAHE contrast enhancement, mask binarization
5. **Stratified splits** -- 70/15/15 train/val/test, stratified by class and ITA skin tone group, zero data leakage verified

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Deep Learning | PyTorch | 2.10+ |
| Vision Backbone | DINOv2 (Meta) | ViT-B/14 |
| Medical Imaging | MONAI | 1.5+ |
| Segmentation | Segmentation Models PyTorch | 0.5+ |
| Augmentation | Albumentations | 1.4+ |
| Data Quality | CleanVision, Cleanlab | Latest |
| API | FastAPI | 0.115+ |
| Inference | ONNX Runtime | 1.20+ |
| Frontend | Next.js 16, React 19, MUI 7 | Latest |
| Compute | Northeastern Explorer HPC (A100/H200 GPUs) | -- |

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

### Training (DINOv2)

```bash
# Phase 1: Linear probe (frozen backbone)
python scripts/train.py --task classify --backbone dinov2_vitb14 --epochs 10 --lr 1e-3 --device cuda
python scripts/train.py --task segment --backbone dinov2_vitb14 --epochs 10 --lr 1e-3 --device cuda

# Phase 2: LoRA fine-tuning
python scripts/train.py --task classify --backbone dinov2_vitb14 --epochs 30 --lr 5e-5 --use-lora --device cuda
python scripts/train.py --task segment --backbone dinov2_vitb14 --epochs 30 --lr 5e-5 --use-lora --device cuda

# Legacy U-Net++ baseline (for comparison)
python scripts/train.py --task segment-unetpp --epochs 100 --device cuda
```

### HPC Training (SLURM)

```bash
sbatch slurm/train_dinov2.sh              # Full pipeline (Phase 1 + Phase 2)
sbatch slurm/train_dinov2.sh classify     # Classifier only
sbatch slurm/train_dinov2.sh segment      # Segmenter only
```

### Evaluation

```bash
# Evaluate DINOv2 classifier
python scripts/evaluate.py --task classify \
    --checkpoint checkpoints/dinov2_classifier/best_epoch009_0.9785.pt \
    --backbone dinov2_vitb14 --device cuda

# Evaluate DINOv2 segmenter
python scripts/evaluate.py --task segment \
    --checkpoint checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt \
    --backbone dinov2_vitb14 --device cuda
```

### FastAPI Server

```bash
cd DiaFoot.AI
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export DIAFOOT_CLASSIFIER_CKPT=checkpoints/dinov2_classifier/best_epoch009_0.9785.pt
export DIAFOOT_SEGMENTER_CKPT=checkpoints/dinov2_segmenter/best_epoch009_0.1062.pt
export DIAFOOT_DEVICE=cpu
uvicorn src.deploy.app:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3000
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
│   ├── models/                 # DINOv2 classifier/segmenter, U-Net++, FUSegNet
│   ├── training/               # Trainer, losses, schedulers
│   ├── evaluation/             # Metrics, fairness, calibration, robustness
│   ├── inference/              # Pipeline, TTA, postprocessing
│   └── deploy/                 # FastAPI app
├── scripts/                    # Entry points for train, eval, export
├── slurm/                      # HPC job scripts
├── frontend/                   # Next.js React UI
├── results/                    # Metrics, figures, reports
├── checkpoints/                # Trained model weights
└── tests/                      # Unit tests
```

---

## Honest Limitations

1. **Classifier may still learn dataset shortcuts.** While DINOv2 achieves 98.36% test accuracy (vs EfficientNet's 100%), the three data categories come from visually distinct sources. External validation is needed to confirm generalization. A production system requires same-source data across all classes.

2. **Segmentation slightly below fully fine-tuned baseline.** DINOv2 frozen backbone achieves 82.73% Dice vs U-Net++'s 85.89%, but with 10x fewer training epochs and 0.2% trainable parameters. Further LoRA fine-tuning or full backbone unfreezing could close this gap.

3. **Wagner staging was not trained.** The architecture supports it, but clinical grade labels were unavailable. This requires clinical partnerships.

4. **Limited skin tone diversity.** The dataset is predominantly a single ITA group. Fairness conclusions cannot be generalized to the full Fitzpatrick I-VI spectrum.

5. **Not validated on standardized benchmarks.** Results are on our own data splits. Comparison against the DFUC 2022 challenge leaderboard would require access to their test set.

---

## Regulatory & Ethical Notice

This project is developed **strictly for academic and educational purposes**. It is part of the AAI6620 Computer Vision coursework at Northeastern University.

**This software:**
- Is **NOT** a medical device as defined by the FDA, EU MDR, or any regulatory body
- Has **NOT** undergone clinical validation, regulatory review, or approval of any kind
- Is **NOT** intended to diagnose, treat, cure, or prevent any disease or medical condition
- Should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment
- Makes **NO** claims of clinical accuracy, safety, or efficacy

**If you are experiencing a medical emergency or have concerns about a diabetic foot ulcer, contact your healthcare provider immediately.**

---

## Citation

```
@misc{bandari2026diafoot,
  title={DiaFoot.AI: A Multi-Task Pipeline for Diabetic Foot Ulcer Detection and Segmentation with DINOv2 Transfer Learning},
  author={Bandari, Ruthvik},
  year={2026},
  institution={Northeastern University},
  course={AAI6620 Computer Vision},
  note={Academic project -- not for clinical use}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

This license grants permission for academic and research use. **It does not grant permission for clinical or diagnostic use.**

---

*Built with care for educational impact. Data composition matters more than architecture.*
