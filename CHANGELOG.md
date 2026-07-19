# Changelog

All notable changes to DiaFoot.AI are documented in this file.

## [2.1.0] — 2026-07-19

### Training-data composition study + reproducible release

Adds a controlled study of how training-data *composition* affects DFU segmentation, plus honest
clean-split re-evaluation and public release artifacts. (The v2.0.0 headline Dice of 85.89% below is a
DFU-only, single-skin-tone subgroup figure that predates the data-leakage fix — see `README.md` and
`docs/PROJECT_REPORT.md` for the honest, whole-test-set numbers.)

### Added
- **Training-data composition study** — 5 compositions × 3 architectures (U-Net++, SegFormer-B0,
  DINOv2) × 5-fold cross-validation (75 models). Finding, identical across all three architectures:
  DFU+Healthy > DFU-only > All > DFU+Non-DFU > Random-mixed — *composition beats size*. A size-matched
  Random-mixed control isolates composition from dataset size.
- Composition harness: `src/data/composition.py`, `scripts/run_composition_experiment.py`,
  `scripts/aggregate_composition_results.py`, `scripts/make_cv_folds.py`,
  `src/evaluation/composition_report.py` (bootstrap CIs, FP-on-empty, paired bootstrap, provenance
  hashes), and `docs/COMPOSITION_EXPERIMENT_RUNBOOK.md`.
- Public releases (HuggingFace): per-cell model scores + model card
  (`RuthvikBandari/dfu-segmentation-composition`) and a reproducible dataset benchmark
  (`RuthvikBandari/dfu-segmentation-benchmark` — manifest, license-permitted masks, splits/CV, source
  pointers). Both private pending paper submission.
- Manuscript in preparation: "Beyond Bigger Datasets: How Training Data Composition Influences Diabetic
  Foot Ulcer Segmentation" (SPIE Medical Imaging), with Prof. Mohammad Eslami (Harvard Medical School).

### Changed
- `README.md` and `docs/PROJECT_REPORT.md` (§5.3) updated with the all-fold composition results; the
  dashboard composition card now shows the honest study (replacing the pre-clean-split ablation numbers).
- Data sourcing clarified: four public datasets (FUSeg, AZH, Mendeley `hsj38fwnvr`, Kaggle); the single
  Mendeley record supplies both healthy (`Normal`) and non-DFU (`Wound_Main`) images.

### Fixed
- Reported segmentation metrics are on leakage-audited clean splits (near-duplicate train↔test leakage
  reduced from 96,829 pairs to 0), superseding earlier leakage-inflated figures.

## [2.0.0] — 2026-03-05

### Complete Rebuild: v1 → v2

DiaFoot.AI v2 is a ground-up rebuild addressing fundamental flaws in v1's data pipeline and evaluation methodology.

### Why the Rebuild

v1 achieved 84.93% IoU / 91.73% Dice but had zero clinical specificity — it predicted ulcers on every input because it was trained exclusively on ulcer images with no concept of "not a wound." v2 transforms the project from a toy segmentation model into a clinically meaningful multi-task pipeline.

### Added

**Data Foundation**
- Three-category dataset: 2,119 DFU + 3,300 healthy + 2,686 non-DFU = 8,105 total images
- Production data cleaning pipeline: integrity checks, mask validation, perceptual hash deduplication
- CleanVision quality audit on all raw data (blur, darkness, duplicates, low-information detection)
- ITA skin tone analysis for fairness evaluation
- Stratified train/val/test splits (70/15/15) with zero leakage verification
- AZH wound care center dataset integration (1,109 new wound images)

**Architecture**
- Multi-task cascaded pipeline: Triage Classifier → Wound Segmenter
- EfficientNet-V2-M triage classifier (healthy vs non-DFU vs DFU)
- U-Net++ with EfficientNet-B4 encoder and scSE attention
- FUSegNet with EfficientNet-B7 and P-scSE attention (comparison architecture)
- MedSAM2 LoRA fine-tuning setup (implemented, not trained)
- nnU-Net v2 wrapper (implemented, not trained)

**Training**
- DiceCE compound loss function
- Cosine annealing scheduler with linear warmup
- Exponential Moving Average (EMA) weight tracking
- Early stopping (patience=15 epochs)
- BFloat16 mixed precision on H200 GPUs
- SLURM array jobs for parallel ablation studies

**Evaluation**
- Full metrics suite: Dice, IoU, HD95, NSD@2mm, NSD@5mm, ASSD
- Clinical metrics: wound area estimation (mm²), wound perimeter
- Test-Time Augmentation (TTA) with 16 augmentations (+3.88% Dice improvement)
- ITA-stratified fairness audit (0.00% gap on DFU images)
- Data composition ablation (DFU-only vs mixed training)
- Architecture ablation (U-Net++ vs FUSegNet)

**Deployment**
- End-to-end inference pipeline (classify → segment → measure)
- FastAPI REST API with /predict endpoint
- ONNX export pipeline with validation and benchmarking
- Prediction visualization with mask overlays

**Documentation**
- Comprehensive README with results tables and honest limitations
- Dataset card documenting all sources and ITA distribution
- 38-commit structured development plan mapped to peer feedback
- Results notebooks with publication-ready figures

### Changed

- Training data: ulcer-only → three categories (healthy + non-DFU + DFU)
- Architecture: single binary segmenter → cascaded multi-task pipeline
- Evaluation: Dice/IoU only → Dice, IoU, HD95, NSD, clinical metrics, fairness
- Loss function: Focal Tversky → DiceCE compound loss
- Encoder: EfficientNet-B4 → EfficientNet-B4 with scSE attention
- Training loop: basic loop → production trainer with EMA, early stopping, checkpointing
- Data pipeline: raw images → cleaned, validated, deduplicated, stratified

### Fixed

- Model predicting ulcers on healthy skin (added negative examples)
- No data quality assurance (added CleanVision + Cleanlab pipeline)
- Inflated metrics from training on uncleaned data
- No skin tone fairness analysis (added ITA-stratified evaluation)
- No boundary quality metrics (added HD95, NSD)

### Key Results

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Dice (overall) | 91.73%* | 85.89% | Honest measurement |
| IoU (overall) | 84.93%* | 79.35% | Honest measurement |
| Clinical specificity | 0% | 100% (classifier) | From useless to useful |
| HD95 | Not measured | 17.3 px | New metric |
| NSD@5mm | Not measured | 94.74% | New metric |
| Wound area error | Not measured | 1.1% | New metric |
| Skin tone fairness gap | Not measured | 0.00% | New metric |
| Data categories | 1 (DFU only) | 3 (DFU + healthy + non-DFU) | Clinically complete |
| Data cleaning | None | Full pipeline | Production grade |

*\*v1 metrics were inflated by training on uncleaned data without negative examples.*

---

## [1.0.0] — 2026-02-01

### Initial Release

- U-Net++ with EfficientNet-B4 encoder
- FUSeg dataset (1,210 images, ulcers only)
- Focal Tversky loss
- 84.93% IoU, 91.73% Dice
- Basic CLAHE preprocessing
- No data cleaning, no negative examples, no fairness analysis
