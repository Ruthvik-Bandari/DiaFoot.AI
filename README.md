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
- **Strong transfer** -- Achieves 89.12% Dice on wound segmentation with just 10 epochs on a frozen backbone
- **Better calibration** -- ECE of 0.0075 after temperature scaling enables reliable clinical deferral

### Why Cascaded?

The data composition ablation proved that the segmenter performs best when trained exclusively on DFU images (85.13% Dice with U-Net++). Adding non-DFU wounds hurt performance (79.03% Dice). The classifier handles triage; the segmenter focuses on DFU morphology.

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
| **Dice Score** | **89.12%** | 85.13% |
| **IoU (Jaccard)** | **82.87%** | 77.51% |
| **HD95** | **11.32 px** | -- |
| **NSD@2mm** | **88.80%** | -- |
| **NSD@5mm** | **97.23%** | 95.23% |
| **Wound Area** | 1,274 mm2 predicted vs 1,270 mm2 GT | -- |

Dice 95% CI: [87.07%, 90.96%]; IoU 95% CI: [80.77%, 84.87%]

### External Segmentation Validation (n=552)

The segmenter generalizes well to unseen external data:

| Metric | Internal (n=263) | External (n=552) | Drop |
|--------|-----------------|-------------------|------|
| Dice | 89.12% | **89.29%** | -0.17% (improved) |
| IoU | 82.87% | **82.94%** | -0.06% (improved) |
| HD95 | 11.32 px | **9.51 px** | +1.80 (improved) |

### 5-Fold Cross-Validation (U-Net++ Segmentation)

| Fold | Dice | IoU | n_val |
|------|------|-----|-------|
| 0 | 84.68% | 77.70% | 371 |
| 1 | 85.94% | 79.30% | 371 |
| 2 | 86.63% | 79.71% | 371 |
| 3 | 84.83% | 78.01% | 371 |
| 4 | 84.56% | 77.69% | 372 |
| **Mean +/- Std** | **85.33 +/- 0.91%** | **78.48 +/- 0.95%** | -- |

### Data Composition Ablation

The single most important experiment -- proving that data composition matters more than architecture:

| Training Data | Dice | IoU | NSD@5mm |
|---------------|------|-----|---------|
| **U-Net++ (DFU-only)** | **85.13%** | **77.51%** | **95.23%** |
| U-Net++ (All classes) | 82.35% | 73.67% | 93.34% |
| FUSegNet (DFU+nonDFU) | 81.75% | 73.00% | 92.28% |
| U-Net++ v2 (DFU+nonDFU, fixed) | 80.39% | 70.72% | 90.80% |
| U-Net++ (DFU+nonDFU) | 79.03% | 69.03% | 91.18% |

### Training Efficiency

| Metric | DINOv2 | Previous Models |
|--------|--------|----------------|
| Trainable params (classifier) | 199K (0.2%) | 54M (100%) |
| Trainable params (segmenter) | 11.6M (11.8%) | 25M (100%) |
| Classifier training time | 6 min (10 epochs) | 30+ min (50 epochs) |
| Segmenter training time | 2 min (10 epochs) | 90+ min (90 epochs) |

### Training Setup

| Parameter | Classifier | Segmenter |
|-----------|-----------|-----------|
| Optimizer | AdamW | AdamW |
| LR (Phase 1 / Phase 2) | 1e-3 / 5e-5 | 1e-3 / 5e-5 |
| Batch size | 16 | 16 |
| Epochs (Phase 1 / Phase 2) | 10 / 30 | 10 / 30 |
| Loss | FocalLoss (gamma=2.0) | DiceCELoss |
| Scheduler | Cosine + 5 warmup epochs | Cosine + 5 warmup epochs |
| Precision | bf16-mixed | bf16-mixed |
| GPU | NVIDIA A100 (1 GPU) | NVIDIA A100 (1 GPU) |

### ONNX Export & Deployment

| Metric | Value |
|--------|-------|
| Parity (max abs diff) | 0.00569 |
| Mask agreement (min) | 99.9996% |
| PyTorch latency | 1,055.8 ms |
| ONNX latency | 235.0 ms |
| **Speedup** | **4.5x** |
| File size | 1.8 MB (architecture) + 78 MB (weights) |

### Shortcut Audit

| Perturbation | Baseline Acc | Perturbed Acc | Drop |
|-------------|-------------|--------------|------|
| noise_border | 100% | 96.55% | 3.45% |
| center_only | 100% | 100% | 0.0% |
| blur_background | 100% | 99.69% | 0.31% |

The classifier relies on center content and is robust to background blur.

### Fairness Analysis (ITA-Stratified, DFU-Only)

| ITA Group | Count | Dice | IoU | HD95 | NSD@2mm |
|-----------|-------|------|-----|------|---------|
| Brown | 285 | 85.89% | 79.35% | 17.3 | 85.86% |
| **Fairness gap** | -- | **0.00%** | -- | -- | -- |
| **Bias concern** | -- | **false** | -- | -- | -- |

**Limitation:** The dataset is predominantly composed of a single ITA skin tone group (929/1,057 test images labeled "Unknown" ITA). The model has not been validated across the full Fitzpatrick I-VI spectrum.

---

## Dataset

### Composition (8,105 processed images, 6,996 in final splits)

| Category | Processed | In Splits | Purpose |
|----------|-----------|-----------|---------|
| **DFU** | 2,119 | 1,010 | Wound segmentation training (FUSeg + AZH) |
| **Healthy Feet** | 3,300 | 3,300 | True negatives for classifier |
| **Non-DFU Conditions** | 2,686 | 2,686 | Hard negatives (general wounds, not DFU) |

### Sources

| Dataset | Images | Type |
|---------|--------|------|
| FUSeg 2021 (UWM BigData Lab) | 1,010 | DFU with segmentation masks |
| AZH Wound Care Center | 1,109 | Clinical wound patches with masks |
| Kaggle DFU Patches | 543 | Healthy foot skin patches |
| Mendeley Wound Dataset (Normal) | 2,757 | Healthy foot images |
| Mendeley Wound Dataset (Wounds) | 2,686 | Non-DFU wound images with masks |

### Train/Val/Test Splits

| Split | Total | DFU | Healthy | Non-DFU |
|-------|-------|-----|---------|---------|
| Train | 4,894 (70%) | 704 | 2,310 | 1,880 |
| Val | 1,045 (15%) | 148 | 495 | 402 |
| Test | 1,057 (15%) | 158 | 495 | 404 |

### Data Pipeline

1. **Integrity check** -- verify every image opens and is not corrupt
2. **Mask validation** -- binary format check, dimension alignment, coverage statistics
3. **Deduplication** -- perceptual hash (dHash) to remove cross-dataset duplicates
4. **Preprocessing** -- resize to 512x512 (aspect-preserving pad), CLAHE contrast enhancement, mask binarization
5. **Stratified splits** -- 70/15/15 train/val/test, stratified by class and ITA skin tone group

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

1. **Classifier learns dataset shortcuts -- confirmed by external validation.** The EfficientNet classifier achieves 100% internal accuracy but only 21% on external data, with 0% DFU sensitivity. The DINOv2 classifier (98.36%) is less extreme but the three data categories come from visually distinct sources (different cameras, backgrounds). A production system requires same-source data across all classes.

2. **Data leakage detected in healthy feet.** The leakage audit found 20,774 near-duplicate pairs (perceptual hash distance=0) across train-val splits, primarily in Kaggle healthy foot patches. Content overlap: 87 train-val, 9 train-test pairs. This inflates healthy class metrics.

3. **Limited skin tone diversity.** 929 of 1,057 test images are labeled "Unknown" ITA. The fairness audit covers only a single ITA group (Brown, n=285 DFU). Conclusions cannot be generalized to the full Fitzpatrick I-VI spectrum.

4. **Wound area agreement evaluated on only 3 images** (MAE: 7.23 mm2, Pearson r: 0.9997). This is statistically insufficient for clinical claims despite high correlation.

5. **Wagner staging was not trained.** The architecture supports it, but clinical-grade labels were unavailable. This requires clinical partnerships.

6. **Not validated on standardized benchmarks.** Results are on our own data splits. Comparison against the DFUC 2022 challenge leaderboard would require access to their test set.

7. **Patient overlap across splits.** The data pipeline report shows 22 train-val, 15 train-test, and 4 val-test patient overlaps.

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
