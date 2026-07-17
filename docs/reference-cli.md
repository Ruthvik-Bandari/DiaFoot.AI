# CLI reference

This page lists every command-line script that ships with DiaFoot.AI, grouped by what
you use it for. The four core-workflow scripts (`train.py`, `evaluate.py`, `predict.py`,
`export_onnx.py`) get full flag tables pulled straight from their argparse definitions.
The data-pipeline and experiment scripts get a one-line purpose and a working example each.
All commands run from the repository root and assume the `diafootai` package is installed
(`pip install -e .`). For headline performance numbers, see the results section in
[../README.md](../README.md). For step-by-step walkthroughs, see the how-to guides linked
under [Related](#related).

---

## 1. Core workflow

These four scripts cover the everyday loop: train a model, evaluate a checkpoint, predict on
a new image, and export to ONNX. Two of them also install as console commands (`diafootai-train`,
`diafootai-eval`) — see [section 4](#4-installed-console-scripts).

### `scripts/train.py`

Train a DINOv2 classifier, a DINOv2 segmenter, or the legacy U-Net++ segmenter. The classifier
trains on all three classes; the segmenters train on DFU images only.

| Flag | Type | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `--task` | str | (none) | Yes | Training mode: `classify`, `segment`, or `segment-unetpp`. |
| `--config` | str | `configs/training/dinov2_baseline.yaml` | No | Path to the training config YAML. |
| `--splits-dir` | str | `data/splits` | No | Directory holding `train.csv` / `val.csv`. |
| `--device` | str | `cuda` | No | Torch device to train on. |
| `--epochs` | int | `50` | No | Number of training epochs. |
| `--batch-size` | int | `16` | No | Mini-batch size. |
| `--num-workers` | int | `8` | No | DataLoader worker processes. |
| `--verbose` | flag | off | No | Enable DEBUG-level logging. |
| `--backbone` | str | `dinov2_vitb14` | No | DINOv2 backbone: `dinov2_vits14`, `dinov2_vitb14`, or `dinov2_vitl14`. |
| `--lr` | float | `1e-4` | No | Learning rate. |
| `--weight-decay` | float | `1e-2` | No | AdamW weight decay. |
| `--use-lora` | flag | off | No | Apply LoRA adapters to backbone attention layers. |
| `--lora-rank` | int | `8` | No | LoRA rank. |
| `--lora-alpha` | int | `16` | No | LoRA alpha scaling. |
| `--unfreeze-backbone` | flag | off | No | Unfreeze the backbone for full fine-tuning (Phase 3). |

```bash
# Train the triage classifier
python scripts/train.py --task classify --device cuda --epochs 50

# Train the DINOv2 wound segmenter with LoRA adapters
python scripts/train.py --task segment --use-lora --backbone dinov2_vitb14
```

### `scripts/evaluate.py`

Evaluate a trained classifier or segmenter on the test split and write metrics JSON under
`results/`.

| Flag | Type | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `--task` | str | (none) | Yes | `classify` or `segment`. |
| `--checkpoint` | str | (none) | Yes | Path to the model checkpoint (`.pt`). |
| `--splits-dir` | str | `data/splits` | No | Directory holding `test.csv`. |
| `--device` | str | `cuda` | No | Torch device (falls back to CPU if CUDA is unavailable). |
| `--verbose` | flag | off | No | Enable DEBUG-level logging. |
| `--model` | str | `dinov2` | No | Model type: `dinov2` or `unetpp` (legacy). |
| `--backbone` | str | `dinov2_vitb14` | No | DINOv2 backbone: `dinov2_vits14`, `dinov2_vitb14`, or `dinov2_vitl14`. |

```bash
# Evaluate the DINOv2 classifier
python scripts/evaluate.py --task classify \
    --checkpoint checkpoints/dinov2_classifier/best.pt

# Evaluate the DINOv2 segmenter
python scripts/evaluate.py --task segment \
    --checkpoint checkpoints/dinov2_segmenter/best.pt
```

### `scripts/predict.py`

Run the cascaded pipeline on a single foot image: classify, then segment if a wound is
suspected, and print the estimated wound area.

| Flag | Type | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `--image` | str | (none) | Yes | Path to the foot image. |
| `--classifier-checkpoint` | str | `checkpoints/dinov2_classifier/best.pt` | No | Classifier checkpoint. |
| `--segmenter-checkpoint` | str | `checkpoints/dinov2_segmenter/best.pt` | No | Segmenter checkpoint. |
| `--backbone` | str | `dinov2_vitb14` | No | DINOv2 backbone: `dinov2_vits14`, `dinov2_vitb14`, or `dinov2_vitl14`. |
| `--save-mask` | str | `None` | No | Path to save the predicted mask PNG (skipped if omitted). |
| `--device` | str | `cpu` | No | Torch device. |

```bash
# Classify and segment one image
python scripts/predict.py --image path/to/foot_image.jpg

# Save the predicted wound mask
python scripts/predict.py --image path/to/image.jpg --save-mask output_mask.png
```

### `scripts/export_onnx.py`

Export a trained checkpoint to ONNX for production inference, then optionally validate parity
against PyTorch and benchmark inference speed.

| Flag | Type | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `--checkpoint` | str | (none) | Yes | Path to the model checkpoint. |
| `--output` | str | `models/diafoot_segmenter.onnx` | No | Output ONNX path. |
| `--model` | str | `dinov2` | No | Model type: `dinov2`, `dinov2-classifier`, or `unetpp`. |
| `--backbone` | str | `dinov2_vitb14` | No | DINOv2 backbone: `dinov2_vits14`, `dinov2_vitb14`, or `dinov2_vitl14`. |
| `--validate` | flag | off | No | Check that ONNX outputs match PyTorch within tolerance. |
| `--benchmark` | flag | off | No | Measure ONNX inference speed and write `onnx_benchmark.json`. |
| `--verbose` | flag | off | No | Enable DEBUG-level logging. |

Input size is `518x518` for DINOv2 models and `512x512` for U-Net++.

```bash
# Export and validate the DINOv2 segmenter
python scripts/export_onnx.py \
    --checkpoint checkpoints/dinov2_segmenter/best.pt \
    --model dinov2 --output models/diafoot_segmenter.onnx \
    --validate --verbose
```

---

## 2. Data pipeline and preparation

These scripts collect, clean, label, preprocess, and split the dataset. `run_data_pipeline.py`
orchestrates the end-to-end sequence; the others let you run individual stages. For the full
walkthrough, see [howto-run-data-pipeline.md](howto-run-data-pipeline.md).

### `scripts/collect_healthy_feet.py`

Download, organize, validate, and audit healthy foot images into `data/raw/healthy`.

```bash
python scripts/collect_healthy_feet.py --verbose
```

### `scripts/collect_non_dfu.py`

Collect hard-negative non-DFU wound images (Mendeley wounds, optional DermNet) into
`data/raw/non_dfu`.

```bash
python scripts/collect_non_dfu.py --audit
```

### `scripts/integrate_azh_data.py`

Clean and integrate the AZH wound-care dataset into `processed/dfu` (integrity check, mask
validation, deduplication, resize to 512x512, CLAHE, mask binarization).

```bash
python scripts/integrate_azh_data.py --verbose
```

### `scripts/run_cleaning.py`

Run a CleanVision quality audit on downloaded datasets and write findings to `data/metadata`.

```bash
python scripts/run_cleaning.py --data-dir data/raw/dfu/fuseg --name fuseg
```

### `scripts/run_label_audit.py`

Audit label quality and set up Wagner-grade labeling metadata.

```bash
python scripts/run_label_audit.py --verbose
```

### `scripts/run_ita_analysis.py`

Compute ITA skin-tone angles and skin-tone group metadata for every image, writing CSV and a
report JSON to `data/metadata`.

```bash
python scripts/run_ita_analysis.py --verbose
```

### `scripts/run_preprocessing.py`

Preprocess raw images (resize, CLAHE, mask binarize) into `data/processed` and build the
stratified splits.

```bash
python scripts/run_preprocessing.py --target-size 512
```

### `scripts/rebuild_splits_strict.py`

Rebuild train/val/test splits with stricter near-duplicate leakage control, assigning full
near-duplicate components to a single split.

```bash
python scripts/rebuild_splits_strict.py --split-dir data/splits --near-threshold 6
```

### `scripts/run_leakage_audit.py`

Audit existing splits for path, canonical-id, content-hash, and near-duplicate leakage, and
write a leakage report JSON.

```bash
python scripts/run_leakage_audit.py \
    --splits-dir data/splits \
    --output data/metadata/leakage_report.json
```

### `scripts/run_data_pipeline.py`

Run the full pipeline end to end: integrate AZH data, validate processed pairs, regenerate
stratified 70/15/15 splits, check for leakage, and write a pipeline report.

| Flag | Type | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `--data-dir` | str | `data` | No | Root data directory. |
| `--seed` | int | `42` | No | Random seed for the split. |
| `--verbose` | flag | off | No | Enable DEBUG-level logging. |

```bash
python scripts/run_data_pipeline.py --verbose
```

### `scripts/build_external_segmentation_split.py`

Build an external DFU segmentation holdout with real masks (default source: the FUSeg
validation set), excluding stems that appear in the internal splits.

```bash
python scripts/build_external_segmentation_split.py \
    --output data/splits/external_segmentation.csv
```

### `scripts/build_external_split_from_patches.py`

Build an external classification holdout from DFU patch folders, removing any exact-content
overlap with the internal splits using SHA256 hashes.

```bash
python scripts/build_external_split_from_patches.py --output data/splits/external.csv
```

---

## 3. Evaluation and experiments

These scripts run deeper analyses: cross-validation, ablations, fairness and subgroup audits,
external validation, robustness checks, and figure generation. Most write a JSON report under
`results/`.

### `scripts/evaluate_all.py`

Evaluate U-Net++, FUSegNet, and ablation checkpoints in a single run.

```bash
python scripts/evaluate_all.py --device cuda
```

### `scripts/evaluate_baseline_split.py`

Evaluate a baseline U-Net++ checkpoint on a chosen split and class subset.

```bash
python scripts/evaluate_baseline_split.py \
    --checkpoint checkpoints/unetpp_baseline/best.pt \
    --split test --include-classes dfu,non_dfu
```

### `scripts/run_cross_val.py`

Train U-Net++ segmentation on one of five cross-validation folds (run once per fold, often as
a SLURM array job).

```bash
python scripts/run_cross_val.py --fold 0 --device cuda --epochs 50
```

### `scripts/summarize_cv_results.py`

Aggregate per-fold cross-validation JSON outputs into a mean and standard-deviation summary.

```bash
python scripts/summarize_cv_results.py --results-dir results --folds 5
```

### `scripts/run_ablation.py`

Run the data-composition ablation: train a segmenter on `dfu_only`, `dfu_nondfu`, or `all`
classes to measure the effect of adding negatives.

```bash
python scripts/run_ablation.py --variant dfu_nondfu --device cuda --epochs 50
```

### `scripts/run_fairness_audit.py`

Connect ITA skin-tone scores to actual model predictions and report fairness metrics across
skin-tone groups.

```bash
python scripts/run_fairness_audit.py \
    --cls-checkpoint checkpoints/dinov2_classifier/best.pt \
    --seg-checkpoint checkpoints/dinov2_segmenter/best.pt
```

### `scripts/run_subgroup_audit.py`

Run a subgroup performance audit across ITA, source dataset, and wound size, with confidence
intervals.

```bash
python scripts/run_subgroup_audit.py \
    --cls-checkpoint checkpoints/dinov2_classifier/best.pt \
    --seg-checkpoint checkpoints/dinov2_segmenter/best.pt
```

### `scripts/run_area_agreement.py`

Evaluate agreement between model-predicted and manual wound-area measurements from a CSV.

```bash
python scripts/run_area_agreement.py --input-csv data/metadata/area_pairs.csv
```

### `scripts/run_external_validation.py`

Compare performance on the internal test split against an external holdout and report the
generalization drop.

```bash
python scripts/run_external_validation.py \
    --external-split data/splits/external.csv \
    --cls-checkpoint checkpoints/dinov2_classifier/best.pt \
    --seg-checkpoint checkpoints/dinov2_segmenter/best.pt
```

### `scripts/run_failure_atlas.py`

Generate a failure atlas of the worst segmentation predictions for qualitative review.

```bash
python scripts/run_failure_atlas.py \
    --checkpoint checkpoints/dinov2_segmenter/best.pt --top-k 30
```

### `scripts/run_shortcut_audit.py`

Perturb image backgrounds and borders to quantify how much the classifier relies on
non-clinical shortcut cues.

```bash
python scripts/run_shortcut_audit.py \
    --checkpoint checkpoints/dinov2_classifier/best.pt --max-samples 1000
```

### `scripts/run_tta_eval.py`

Compare segmentation performance with and without test-time augmentation.

```bash
python scripts/run_tta_eval.py \
    --checkpoint checkpoints/dinov2_segmenter/best.pt --device cuda
```

### `scripts/run_onnx_parity.py`

Benchmark PyTorch versus ONNXRuntime parity on a split and report the maximum output
difference.

```bash
python scripts/run_onnx_parity.py \
    --checkpoint checkpoints/dinov2_segmenter/best.pt \
    --onnx models/diafoot_segmenter.onnx
```

### `scripts/run_repro_bundle.py`

Generate a reproducibility bundle (environment, config, and result snapshot JSON).

```bash
python scripts/run_repro_bundle.py --output results/repro/repro_bundle.json
```

### `scripts/visualize_results.py`

Generate overlay figures comparing predictions against ground truth for the report.

```bash
python scripts/visualize_results.py \
    --checkpoint checkpoints/dinov2_segmenter/best.pt --num-images 10 --device cpu
```

### `scripts/train_classifier.py`

Train the triage classifier from a YAML config (a lighter alternative to `train.py --task classify`).

| Flag | Type | Default | Required? | Description |
| --- | --- | --- | --- | --- |
| `--config` | str | `configs/model/classifier.yaml` | No | Classifier config YAML. |
| `--device` | str | `cuda` | No | Torch device. |
| `--seed` | int | `42` | No | Random seed. |

```bash
python scripts/train_classifier.py --config configs/model/classifier.yaml --device cuda
```

### `scripts/generate_midterm_pdf.py`

Render a markdown report to PDF with ReportLab. This is a legacy one-off utility that reads
from hardcoded local paths and takes no command-line flags.

```bash
python scripts/generate_midterm_pdf.py
```

### `scripts/patch_api_mask.py`

Auto-patch `src/deploy/app.py` to add a `segmentation_mask_base64` field to the FastAPI
`/predict` response. Takes no command-line flags.

```bash
python scripts/patch_api_mask.py
```

---

## 4. Installed console scripts

Installing the package (`pip install -e .`) registers two console entry points that map to the
core scripts, so you can call them from anywhere without the `python scripts/...` prefix.

| Command | Maps to | Description |
| --- | --- | --- |
| `diafootai-train` | `scripts.train:main` | Same as `python scripts/train.py`; accepts every flag in the [`train.py` table](#scriptstrainpy). |
| `diafootai-eval` | `scripts.evaluate:main` | Same as `python scripts/evaluate.py`; accepts every flag in the [`evaluate.py` table](#scriptsevaluatepy). |

```bash
# Train the classifier via the console script
diafootai-train --task classify --device cuda --epochs 50

# Evaluate the segmenter via the console script
diafootai-eval --task segment --checkpoint checkpoints/dinov2_segmenter/best.pt
```

---

## Related

- [howto-train.md](howto-train.md) — step-by-step training walkthrough.
- [howto-run-data-pipeline.md](howto-run-data-pipeline.md) — building the dataset and splits from scratch.
- [reference-api.md](reference-api.md) — the REST API reference.
- [../README.md](../README.md) — project overview, install, and results.
