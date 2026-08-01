# DiaFoot.AI: Complete Project Documentation

This is the comprehensive, single-source, self-contained documentation for DiaFoot.AI. Everything is
here, inline: the clinical problem, the cascaded inference pipeline, the dataset and its curation, every
model, the full training-data-composition research study (design, results, mechanism), the full HPC
training pipelines (including the AICR B200 run that produced the published results, with every command
and gotcha), the complete composition-pipeline code reference, the evaluation and audit suite, the
repository layout, the REST service, reproduction, deployment, and limitations.

It is deliberately exhaustive and self-contained: no section defers detail to another file. Shorter,
focused versions of some topics exist as separate docs and are listed at the very end.

---

## Table of contents

1. [What DiaFoot.AI is](#1-what-diafootai-is)
2. [Clinical problem and motivation](#2-clinical-problem-and-motivation)
3. [The cascaded inference pipeline](#3-the-cascaded-inference-pipeline)
4. [Results (honest, clean splits)](#4-results-honest-clean-splits)
5. [The dataset](#5-the-dataset)
6. [Data curation and the leakage fix](#6-data-curation-and-the-leakage-fix)
7. [Models](#7-models)
8. [The training-data-composition study](#8-the-training-data-composition-study)
   - 8.1 [The problem it answers](#81-the-problem-it-answers)
   - 8.2 [The five compositions](#82-the-five-compositions)
   - 8.3 [What is held fixed](#83-what-is-held-fixed)
   - 8.4 [The grid and the metric suite](#84-the-grid-and-the-metric-suite)
   - 8.5 [Full results](#85-full-results)
   - 8.6 [Why composition matters so much](#86-why-composition-matters-so-much)
   - 8.7 [Trade-offs and honest caveats](#87-trade-offs-and-honest-caveats)
   - 8.8 [Alternatives considered](#88-alternatives-considered)
9. [HPC training pipelines (Explorer H200 and AICR B200)](#9-hpc-training-pipelines-explorer-h200-and-aicr-b200)
   - 9.1 [What runs](#91-what-runs)
   - 9.2 [Prerequisites and the clean-split gate](#92-prerequisites-and-the-clean-split-gate)
   - 9.3 [Generate the shared CV folds](#93-generate-the-shared-cv-folds)
   - 9.4 [AICR / MGHPCC (B200): the primary path](#94-aicr--mghpcc-b200-the-primary-path)
   - 9.5 [Northeastern Explorer (H200)](#95-northeastern-explorer-h200)
   - 9.6 [Aggregate, verify, troubleshoot](#96-aggregate-verify-troubleshoot)
   - 9.7 [Other SLURM jobs](#97-other-slurm-jobs)
10. [Composition-pipeline code reference](#10-composition-pipeline-code-reference)
11. [Evaluation and audit suite](#11-evaluation-and-audit-suite)
12. [Repository map](#12-repository-map)
13. [Configuration system](#13-configuration-system)
14. [REST API](#14-rest-api)
15. [Reproduction](#15-reproduction)
16. [Deployment (Docker, ONNX)](#16-deployment-docker-onnx)
17. [Testing](#17-testing)
18. [Limitations](#18-limitations)
19. [The manuscript](#19-the-manuscript)
20. [Focused / shorter documents](#20-focused--shorter-documents)

---

## 1. What DiaFoot.AI is

DiaFoot.AI is a multi-task computer-vision system for diabetic foot images. Given one foot photo it does
three things in sequence: it **triages** the image into `Healthy` / `Non-DFU` / `DFU`, it **segments**
the wound region when one is present, and it **measures** wound area in square millimeters. The package
is `diafootai`, targets Python 3.12 to 3.13 and PyTorch, is MIT-licensed, and is research code, not a
cleared medical device.

The project has two faces:

- A **production-style pipeline** (classifier + segmenter + area estimator) exposed through a FastAPI
  service and a Next.js frontend.
- A **research contribution**: the training-data-composition study (Section 8), which is the subject of
  an SPIE manuscript and the reason for much of the recent code.

A defining principle is honesty about evaluation. All reported numbers come from clean, leakage-audited
splits (Section 6), not cherry-picked subgroups, and the code stamps every result with provenance so it
can be traced back to the exact model and data that produced it.

---

## 2. Clinical problem and motivation

Diabetic foot ulcers are a severe complication of diabetes and a leading cause of nontraumatic
lower-limb amputation. A large fraction of people with diabetes develop a foot ulcer in their lifetime,
and once an ulcer forms the risks of infection, hospitalization, amputation, and death rise sharply;
five-year mortality after a DFU is often compared to that of common cancers. Wound area and its change
over time is a primary indicator of healing, but manual tracing is slow, subjective, and hard to
standardize across clinicians. Automated segmentation from a phone photo offers a consistent, low-cost,
scalable measurement, which is what DiaFoot.AI targets.

---

## 3. The cascaded inference pipeline

A single image flows through `InferencePipeline` (`src/inference/pipeline.py`):

1. **Triage classifier** predicts `Healthy` / `Non-DFU` / `DFU` with calibrated probabilities. Low
   confidence triggers a **defer** ("refer to clinician") rather than a forced label.
2. **Gated segmentation**: only when the wound path is warranted does the segmenter run, producing a
   binary wound mask.
3. **Area and quality**: wound area in mm2 and wound-coverage percentage are computed, plus image
   quality flags (blur, brightness, resolution) so a bad photo is caught early.

The service returns the classification and probabilities, the defer decision and reason, quality flags,
`has_wound`, `wound_area_mm2`, `wound_coverage_pct`, a base64 mask, diagnostics, and timing (Section 14).

---

## 4. Results (honest, clean splits)

Two distinct result sets exist. Do not conflate them, and per project policy never cite the old
`CHANGELOG` v2.0.0 headline (Dice 85.89% / IoU 79.35%): that is a pre-leakage-fix, single-subgroup
number and is a trap. Authoritative numbers live in `results/*.json`.

### 4a. Deployed single-model pipeline (clean splits)

- **Triage classifier**: accuracy about 0.984, macro-F1 about 0.981, DFU sensitivity about 0.966,
  healthy specificity about 0.995, macro-AUROC about 0.999; calibration improves with temperature
  scaling (expected calibration error roughly 0.039 to 0.007).
- **Segmentation**: DFU-only Dice about 0.89, with 5-fold cross-validation Dice about 0.85. On the full
  mixed test set the mean Dice is lower (empty-mask false positives drag the mean down while the median
  stays high).
- **External validation**: segmentation transfers to unseen sources (external DFU Dice stays high), but
  the triage classifier collapses on unseen sources (accuracy near chance, DFU sensitivity near zero).
  This is the headline deployment limitation.
- **Fairness**: the DFU-only segmentation skin-tone gap (by Individual Typology Angle, ITA) is near
  zero; the full mixed-set gap is larger and flagged.
- **Efficiency**: ONNX parity with PyTorch is about 99.99%, and ONNX runs several times faster.

### 4b. Training-data-composition study (75 cells)

The research result. Across three architectures under five-fold cross-validation on one fixed clean
test set, the ordering is identical: **DFU+Healthy > DFU-only > All > DFU+Non-DFU > Random-mixed**. Full
table and interpretation in Section 8.

---

## 5. The dataset

The unified benchmark is 8,105 images across three categories, aggregated from **four** distinct public
datasets (three source families, since FUSeg derives from AZH). Every image has a paired binary mask;
healthy feet carry an all-zero (empty) mask.

| Category | Source dataset | Images | Mask | License |
|---|---|---:|---|---|
| DFU | FUSeg (UW-Milwaukee BigData) | 1,010 | Wound boundary | No explicit data license (permission needed to redistribute masks) |
| DFU | AZH Wound Care Center | 1,109 | Wound boundary | Same source family as FUSeg |
| Healthy feet | Mendeley wound dataset, `Normal` folder | 2,757 | Empty | CC BY 4.0 |
| Healthy feet | Kaggle DFU (normal subset) | 543 | Empty | "Unknown" (no redistribution grant) |
| Non-DFU wounds | Mendeley wound dataset, `Wound_Main` folder | 2,686 | Wound boundary | CC BY 4.0 |

Notes:

- The two Mendeley rows are two folders of a single record (DOI `10.17632/hsj38fwnvr.3`), which is why
  the benchmark is **four** datasets, not five.
- DFUC-2022 was deliberately excluded: it requires a signed data-use agreement and cannot be openly
  redistributed, so including it would block a reproducible release.
- The non-DFU `Wound_Main` pool is not strictly non-diabetic; per its documentation about 16% are
  diabetic-wound images. This is disclosed as a limitation and makes the composition conclusion
  conservative (Section 8.7).
- Dataset construction scripts: `integrate_azh_data.py` (AZH), `collect_healthy_feet.py` (Mendeley +
  Kaggle normals), `collect_non_dfu.py` (Mendeley wounds), orchestrated by `run_data_pipeline.py`.

A public release lives on Hugging Face (private until manuscript submission): masks and splits where the
license permits (Mendeley), with FUSeg/AZH/Kaggle as pointers plus reconstruction instructions.

---

## 6. Data curation and the leakage fix

Curation is treated as a first-class contribution, because the credibility of every result rests on a
clean, leakage-free test set. The pipeline (`src/data/`) runs these stages:

1. **Corrupted images and pairing** (`preprocessing.py`): decode, resize to 512x512, drop orphan
   images/masks.
2. **Mask verification** (`wagner_labeling.py`): binarize masks; assign healthy feet an all-zero mask so
   the model is explicitly supervised where wounds are absent. This matters for the composition study: on
   an empty ground-truth mask any predicted pixel is a false positive, which is exactly how the cost of
   hallucinated wounds is measured.
3. **Exact and perceptual dedup** (`dedup.py`): remove byte-identical duplicates (content hash) and
   near-duplicates (difference hash, dHash).
4. **Leakage removal** (`rebuild_splits_strict.py`, `leakage_audit.py`): rebuild splits so all crops of
   a common source image stay together and near-duplicates are merged before partitioning.
5. **Splits**: a fixed held-out test set (n=1,215, about 15%) stratified by class and skin tone (ITA),
   and a train+val pool (n=6,890; base split 5,674 / 1,216).

The headline of this work: a perceptual-hash audit found **96,829** near-duplicate train-test image
pairs in the naive split, reduced to **zero** after the fix. Earlier "inflated" segmentation numbers
(around 0.98 Dice) were a leakage artifact; the honest re-run brought segmentation to its true level
(around 0.72 mean on the mixed set, about 0.89 DFU-only) while classification stayed genuinely high
(classes are separable).

The audit tool has a known blind spot worth stating: `run_leakage_audit.py` checks path, content-hash,
and dHash near-duplicates, and it can report zero when images are absent (it skips missing files), so it
must be run where the images actually live. Full story:
[HPC_HONEST_RERUN_RUNBOOK.md](HPC_HONEST_RERUN_RUNBOOK.md).

---

## 7. Models

### 7a. Deployed models

- **DINOv2 triage classifier** (`src/models/dinov2_classifier.py`): frozen ViT-B/14 self-supervised
  backbone with a LoRA-capable classification head; two-phase training (linear probe then LoRA).
- **DINOv2 segmenter** (`src/models/dinov2_segmenter.py`, the largest model module): frozen ViT-B/14 +
  a UPerNet decoder (`PPM` + `UPerNetDecoder`). This is the deployed wound segmenter.
- **U-Net++** (`src/models/unetpp.py`): EfficientNet-B4 encoder + scSE decoder attention (`attention.py`
  provides `ChannelSE`, `SpatialSE`, `ParallelScSE`).
- Also present: `fusegnet.py` (FUSegNet baseline), `unetpp_multitask.py` (seg + cls + Wagner staging),
  `nnunet_wrapper.py`, `staging_head.py` (Wagner grading), `boundary_refine.py` (morphological
  post-processing).

### 7b. Study architectures

The composition study (Section 8) sweeps three architecture families, each held fixed across
compositions:

- **U-Net++** (EfficientNet-B4, scSE) — convolutional encoder-decoder.
- **SegFormer-B0** (MiT-B0 encoder, all-MLP decoder) — efficient transformer.
- **DINOv2** (ViT-B/14 frozen + UPerNet) — self-supervised foundation model.

Input size differs by architecture: 512x512 for U-Net++ and SegFormer (divisible by 32), 518x518 for
DINOv2 (divisible by patch size 14).

---

## 8. The training-data-composition study

The flagship research contribution and the subject of the SPIE manuscript (Section 19). Everything about
it is inline below.

### 8.1 The problem it answers

Deep learning teams improve a segmentation model in a predictable way: add more data. When a
disease-specific dataset is small (DFU images are a minority class), the tempting move is to pool every
related labeled image you can find, other wound types, healthy skin, anything with a mask, on the
assumption that a bigger, more varied training set must generalize better.

That assumption hides a confound. Adding images from a different disease or condition does two things at
once: it grows the dataset **and** it shifts the training distribution. If performance changes, you
cannot tell whether the extra *volume* helped or the changed *composition* hurt. Worse, aggregating
public medical datasets brings near-duplicate images everywhere; if any straddle the train/test split
they leak, and a "larger" dataset that happens to place duplicates on both sides can look better purely
from memorization. Most published wound-segmentation work varies the architecture on a fixed dataset.
Almost nobody holds the architecture fixed and varies only what the data contains.

So the question is precise: **does adding heterogeneous wound data improve DFU segmentation, or does
disease-focused training win despite using fewer images?**

### 8.2 The five compositions

Each composition selects a subset (or draw) from the same curated training/validation pool. The held-out
test set is identical for all of them.

| Composition | What it contains | Train images | Role |
|---|---|---:|---|
| `dfu_only` | DFU images only | 1,427 | The disease-focused baseline |
| `dfu_healthy` | DFU + healthy-foot (empty-mask) negatives | 3,645 | In-domain negatives |
| `dfu_nondfu` | DFU + other wound types | 3,279 | Out-of-domain positives |
| `all` | DFU + healthy + non-DFU (everything) | 5,497 | The "more data" condition |
| `random_mixed` | Uniform random draw across all classes, **size-matched to `dfu_only`** | 1,427 | The key control |

`random_mixed` is the linchpin. It uses exactly as many images as `dfu_only` (1,427) but drawn across
all classes, so any difference between the two **cannot** be explained by dataset size. It isolates
composition from volume.

### 8.3 What is held fixed

- **Three architectures**, each frozen across all compositions (Section 7b). Using three families rules
  out "this is an artifact of one network."
- **Training protocol**: AdamW (learning rate 1e-4, weight decay 1e-2), cosine schedule with 5-epoch
  warmup, a compound Dice + cross-entropy loss, bfloat16 mixed precision, gradient clipping, early
  stopping (patience 15) on validation loss, a base seed of 42 offset by the fold index (so a given fold
  uses the same seed for every cell), and a 50-epoch budget. Only the composition changes.
- **Evaluation**: every trained model is scored on the *same fixed clean test set*. DFU-Dice is the
  DFU-positive slice of that identical set, so all cells are judged on the same wounds.

### 8.4 The grid and the metric suite

5 compositions x 3 architectures x 5 cross-validation folds = **75 trained models**. The folds
re-partition the training/validation pool only (they drive early stopping); the test set is never
re-partitioned, so the reported metric is immune to any residual leakage inside the training pool.

A single overlap score hides failure modes, so the study reports several complementary metrics:

- **DFU-Dice** and **IoU** for region overlap (primary quality metric).
- **HD95** (95th-percentile Hausdorff distance, in pixels) and **NSD** (normalized surface Dice) for
  boundary quality. The reported NSD is `nsd_5mm`, which is a 10-pixel tolerance: the source photographs
  are uncalibrated, so `pixel_spacing_mm=0.5` is a nominal constant. It rescales every condition
  identically and cannot change the ranking, but the physical (mm) reading of HD95 and NSD is nominal.
- **FP-on-empty**: the fraction of the 501 test images with an all-zero ground-truth mask (494 healthy
  feet plus 7 DFU-class images whose expert annotation is empty) on which the model predicts any wound
  pixel. With an empty ground truth, any prediction there is a false positive.
  This is the direct measure of a model hallucinating wounds on intact skin, and it is the metric that
  most sharply separates the compositions.

### 8.5 Full results

The ordering is **identical across all three architectures**:

> DFU+Healthy > DFU-only > All > DFU+Non-DFU > Random-mixed

Fold-averaged results (mean over 5 folds; identical held-out test set):

| Arch | Composition | Train | DFU-Dice | IoU | HD95 (median) | NSD | FP-on-empty |
|---|---|---:|---:|---:|---:|---:|---:|
| U-Net++ | DFU-only | 1,427 | 0.872 | 0.799 | 4.09 | 0.948 | 0.083 |
| U-Net++ | **DFU + Healthy** | 3,645 | **0.879** | **0.809** | **4.01** | 0.947 | **0.009** |
| U-Net++ | DFU + Non-DFU | 3,279 | 0.834 | 0.754 | 5.14 | 0.917 | 0.221 |
| U-Net++ | All | 5,497 | 0.846 | 0.770 | 4.79 | 0.923 | 0.015 |
| U-Net++ | Random-mixed | 1,427 | 0.795 | 0.704 | 7.10 | 0.879 | 0.016 |
| SegFormer-B0 | DFU-only | 1,427 | 0.844 | 0.763 | 5.00 | 0.935 | 0.089 |
| SegFormer-B0 | **DFU + Healthy** | 3,645 | **0.856** | **0.778** | **4.42** | **0.943** | **0.008** |
| SegFormer-B0 | DFU + Non-DFU | 3,279 | 0.791 | 0.696 | 7.45 | 0.888 | 0.448 |
| SegFormer-B0 | All | 5,497 | 0.799 | 0.706 | 7.14 | 0.895 | 0.010 |
| SegFormer-B0 | Random-mixed | 1,427 | 0.717 | 0.611 | 11.13 | 0.795 | 0.022 |
| DINOv2 | DFU-only | 1,427 | 0.835 | 0.750 | 5.00 | 0.934 | 0.025 |
| DINOv2 | **DFU + Healthy** | 3,645 | **0.843** | **0.759** | **4.75** | **0.940** | **0.009** |
| DINOv2 | DFU + Non-DFU | 3,279 | 0.792 | 0.697 | 6.73 | 0.903 | 0.058 |
| DINOv2 | All | 5,497 | 0.809 | 0.717 | 6.01 | 0.918 | 0.010 |
| DINOv2 | Random-mixed | 1,427 | 0.752 | 0.645 | 8.82 | 0.861 | 0.019 |

Three findings:

1. **Bigger did not mean better.** `All` (5,497 images) trails `dfu_only` (1,427 images) on every
   architecture and on boundary metrics too. A 3.8x increase in training data came with a *decrease* in
   accuracy.
2. **A fixed image budget is best spent on the target disease.** `random_mixed` uses the same 1,427
   images as `dfu_only` but is the weakest composition throughout, trailing by 8 to 13 Dice points.
   Note the uniform draw leaves it only 374 DFU images against `dfu_only`'s 1,427, so part of that gap
   is the cost of spending the budget off-target; this control alone does not isolate composition from
   size. The clean contrast is finding 1: `All` holds the DFU images fixed at the identical 1,427, adds
   4,070 more, and still loses. Together the two show that neither adding off-target data nor diverting
   budget to it helps.
3. **Other wound types are the harmful ingredient; in-domain negatives are not.** Healthy-foot images
   (same anatomy, empty masks) slightly *improved* DFU-Dice and drove FP-on-empty below 1%. Non-DFU
   wounds *lowered* DFU-Dice and inflated FP-on-empty sharply (up to 44.8% for SegFormer): a model fed
   heterogeneous wounds learns a broad, non-specific notion of "wound" and over-segments healthy skin.

Statistical support: a paired bootstrap on per-image DFU-Dice (pairing within architecture, seed, and
fold) shows every heterogeneous composition below `dfu_only`: negative in all 5 folds, with p < 0.05 in
44 of the 45 fold-level comparisons (the one exception is `unetpp`/`all` in fold 1, p = 0.095), and
p < 0.001 in all 30 `random_mixed` and `dfu_nondfu` comparisons. `dfu_healthy` is the only composition
with a positive mean delta: positive in all 15 fold-level comparisons, significant in 9 of them.

### 8.6 Why composition matters so much

A segmenter trained only on DFU learns a tight prior over what an ulcer looks like and where it appears
on the foot. In-domain healthy-foot images sharpen that prior without diluting it: they share the
anatomy but carry empty masks, so they explicitly teach the model where wounds are *absent* and penalize
spurious activations, which is why they raised Dice slightly and crushed false positives. Non-DFU wounds
pull the other way. Though superficially related, they push the decision boundary toward a generic,
appearance-based notion of "any wound," so the model responds to wound-like color and texture in general
and over-segments both other lesions and healthy skin. The size-matched control makes the mechanism
explicit: diluting a fixed image budget with off-target content is strictly worse than spending all of
it on the target disease.

This has direct clinical weight. In monitoring or triage, the dangerous failure is not a few pixels of
boundary error on a real ulcer, it is a confident wound prediction on intact skin, which erodes trust
and corrupts automated wound-area tracking. That is exactly the failure the non-DFU compositions
amplified and the in-domain negatives suppressed.

### 8.7 Trade-offs and honest caveats

- **Patient-level independence is not guaranteed.** The public sources do not publish real patient IDs,
  so folds are grouped by the strongest available image-provenance identifier, not by patient. Because
  every fold is scored on the fixed clean test set, the reported metric is unaffected by residual
  intra-pool fold leakage, but strict patient independence cannot be claimed.
- **The non-DFU pool is ~16% diabetic.** The non-DFU "hard negative" set is the `Wound_Main` subset of
  the Mendeley source, which per its documentation contains a minority (about 16%) of diabetic-wound
  images. This contamination biases the DFU+Non-DFU comparison *toward* benefiting that composition, yet
  it still underperformed, so the conclusion that non-DFU wounds harm DFU segmentation is conservative.
- **Scope.** The study concerns disease-specific *segmentation*; it may not transfer to detection or
  classification. It spans four public datasets, so generalization to other institutions and cameras
  remains to be established.

### 8.8 Alternatives considered

- **A negative-ratio sweep instead of categorical compositions.** The code supports a dose-response
  curve (`subsample_negatives()`, `--neg-frac`) that keeps every DFU image and a deterministic fraction
  of the negative pool. Demoted to optional/supplementary because the categorical compositions plus the
  size-matched control answer the question more directly.
- **Patient-independent cross-validation.** Impossible here without real patient IDs. The "fixed test set
  every fold" design was chosen precisely so the headline metric is robust to this limitation.
- **A single architecture.** Rejected. Repeating the grid across three architecture families is what
  lets us claim the effect is a property of the data, not one network's inductive bias.

---

## 9. HPC training pipelines (Explorer H200 and AICR B200)

This section runs the full 75-cell study end to end and produces the paper's results table. It documents
both clusters: the AICR B200 path (which produced the published numbers, in full) and the Northeastern
Explorer H200 path. You finish with `results/composition/*.json` (75 provenance-stamped cells) and
`results/composition_comparison.{json,md}` (the fold-averaged table + paired significance).

### 9.1 What runs

75 cells = **5 compositions** x **3 architectures** x **5 CV folds**, packed as a **15-task SLURM array**
(one task per composition x architecture); each task runs its 5 folds sequentially. Each cell trains one
segmentation model on one fold and evaluates it on the fixed clean test set, writing one JSON. Per-cell
JSONs are written as folds finish and the driver skips any cell whose JSON already exists, so an
interrupted or resubmitted array resumes without redoing work.

### 9.2 Prerequisites and the clean-split gate

Prerequisites (both clusters):

- The composition code on the cluster: `scripts/run_composition_experiment.py`,
  `scripts/make_cv_folds.py`, `scripts/aggregate_composition_results.py`, `src/data/composition.py`,
  `src/evaluation/composition_report.py`, and a SLURM array script. Push to GitHub first (you curate
  `main`), then pull on the cluster.
- Clean, leak-free splits and images: `data/splits/{train,val,test}.csv` and
  `data/processed/{dfu,healthy,non_dfu}/{images,masks}/`.
- The canonical clean test split. Its SHA-256 must be
  `d758d68928172c348075a478f7eaa1e496efbd2bf5d51772055126e6ac977851`
  (`sha256sum data/splits/test.csv`). If it differs, you are not on the clean split; stop.

Gate on clean splits (never skip; training on leaky splits reproduces the exact bug this study exists to
avoid):

```bash
python scripts/run_leakage_audit.py --splits-dir data/splits \
    --output data/metadata/leakage_report_composition.json --verbose
python - <<'PY'
import json
r = json.load(open("data/metadata/leakage_report_composition.json"))
flag = r.get("has_any_leakage", r.get("leakage", {}).get("has_any_leakage"))
assert flag is False, f"LEAKAGE PRESENT ({flag}) - STOP, rebuild clean splits first"
print("clean-split gate PASSED: has_any_leakage =", flag)
PY
```

Run this via `srun` on a compute node on clusters whose login node kills heavy Python (Explorer does).
If it fails, rebuild clean splits per [HPC_HONEST_RERUN_RUNBOOK.md](HPC_HONEST_RERUN_RUNBOOK.md).

### 9.3 Generate the shared CV folds

Once, on either cluster:

```bash
python scripts/make_cv_folds.py --n-folds 5 --seed 42
ls data/splits/cv/fold*/train.csv          # expect 5 folds
python -m json.tool data/splits/cv/folds_manifest.json | head -30
```

These patient-grouped folds partition the train+val pool and are shared by every cell, so all
compositions and architectures see the identical fold split. The held-out `test.csv` is never touched.

### 9.4 AICR / MGHPCC (B200): the primary path

AICR is far better suited to this workload than Explorer: the `b200-batch` partition has 224 B200 GPUs
and the QOS allows a per-user cap of 32 GPUs, so **all 15 array tasks run concurrently** and the whole
75-cell matrix finishes in roughly one longest-task (a few hours) instead of one to two days.

Key AICR facts:

- Login: `login.aicr.ai` (host `login0001`, Rocky Linux 9.6). SLURM account: `p2026_0017_neu`.
- Partition `b200-batch`, `--gres=gpu:b200:1`, 24h max wall. QOS `normal`.
- **Compute nodes have internet.** Pretrained weights download at runtime; you do NOT need
  `HF_HUB_OFFLINE` (unlike Explorer).
- **No module system needed** for CUDA: the pinned PyTorch wheel bundles CUDA.
- System Python is 3.9 and there is no conda, so the environment is built with `uv`.

**A1. Build the environment with uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/DiaFoot.AI
uv venv --python 3.13 .venv
source .venv/bin/activate
# Torch verified to run on B200 (CUDA 12.8 wheel). torchaudio is NOT installed:
# its 2.11 pin conflicts with torch 2.8 and it is unused here.
uv pip install torch==2.8.0+cu128 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# The rest mirrors Explorer's pip freeze, minus the cloud-fs pins
# (s3fs / gcsfs / adlfs / aiobotocore) which conflict on fsspec and are unused.
uv pip install -r requirements.txt
python -c "import torch; x=torch.randn(8,8,device='cuda'); print('B200 matmul OK', (x@x).sum().item())"
```

**A2. Stage the data** (about 2.8 GB, ~16k PNGs; both your Mac and Explorer ed25519 keys are authorized
on AICR):

```bash
rsync -av --info=progress2 data/processed/ bandari_ru_neu@login.aicr.ai:~/DiaFoot.AI/data/processed/
rsync -av data/splits/ bandari_ru_neu@login.aicr.ai:~/DiaFoot.AI/data/splits/
# then on AICR, re-verify the clean-split gate (9.2) and regenerate folds (9.3).
```

**A3. Move checkpoints to /scratch BEFORE launching (quota gotcha).** The trainer writes a new
`best_epochNNN.pt` every time the monitored metric improves and never prunes. Across a partial 75-cell
run this reached 83 GB and blew through the ~93 GB `/home` quota, failing 14 of 15 tasks (NFS write
hangs, not always a clean error in the logs). Point `checkpoints/` at `/scratch`:

```bash
rm -rf ~/DiaFoot.AI/checkpoints
mkdir -p /scratch/$USER/diafoot_ckpts
ln -s /scratch/$USER/diafoot_ckpts ~/DiaFoot.AI/checkpoints
quota -s        # confirm /home has headroom; results/ and logs/ stay on /home (small)
```

**A4. Pre-cache the DINOv2 backbone serially BEFORE launching (hub-race gotcha).** `DINOv2Segmenter`
loads its backbone with `torch.hub.load('facebookresearch/dinov2', ...)`, which is not
concurrency-safe. Simultaneous DINOv2 cells race in `~/.cache/torch/hub` (one extracts while another
`rmtree`s the shared dir) and crash with `OSError [Errno 39] Directory not empty`. Warm the cache once,
serially, on the login node:

```bash
python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14'); print('dinov2 cached')"
```

smp models (`unetpp`, `segformer`) use `huggingface_hub`, which is lock-safe, so they are unaffected.

**A5. The AICR SLURM script.** Create `slurm/run_composition_matrix_aicr.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=diafoot-composition
#SBATCH --partition=b200-batch
#SBATCH --account=p2026_0017_neu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00            # 8h backfills; 24h can block (see A6)
#SBATCH --array=0-14               # NO %throttle: all 15 tasks concurrent
#SBATCH --output=logs/slurm/%A_%a_composition.out
#SBATCH --error=logs/slurm/%A_%a_composition.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source .venv/bin/activate
export PYTHONPATH="$PWD"
export NO_ALBUMENTATIONS_UPDATE=1
# NOTE: no HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE here - AICR compute nodes have internet.
mkdir -p logs/slurm results/composition
COMPOSITIONS=(dfu_only dfu_healthy dfu_nondfu all random_mixed)
ARCHS=(unetpp segformer dinov2)
IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCHS[$(( IDX % 3 ))]}
COMP=${COMPOSITIONS[$(( IDX / 3 ))]}
for FOLD in 0 1 2 3 4; do
  RESULT="results/composition/${ARCH}_${COMP}_seed42_fold${FOLD}.json"
  [ -f "$RESULT" ] && { echo "skip existing $RESULT"; continue; }
  QUAL=""; [ "$FOLD" -eq 0 ] && QUAL="--save-qualitative --n-qualitative 8"
  .venv/bin/python scripts/run_composition_experiment.py \
      --arch "$ARCH" --composition "$COMP" --fold "$FOLD" --seed 42 \
      --device cuda --epochs 50 --batch-size 16 --num-workers 16 $QUAL
done
```

**A6. Submit, and use an 8h wall so tasks backfill.**

```bash
sbatch slurm/run_composition_matrix_aicr.sh
squeue -u "$USER"
```

A 24h wall can leave tasks stuck `PD (Priority)` behind a higher-priority job even with idle GPUs,
because a long wall blocks backfill. An 8h wall (plenty for 5 folds on a B200) backfills immediately. If
you already submitted at 24h and tasks are stuck, shorten the wall in place:

```bash
scontrol update job <jobid> TimeLimit=08:00:00     # all 15 tasks backfill onto B200s at once
```

**A7. Monitor and recover DINOv2 cells if they raced.**

```bash
squeue -u "$USER"
ls results/composition/*.json | wc -l        # target: 75
```

If some DINOv2 cells failed fast with the `Errno 39` hub race despite A4, resubmit just those indices
(indices 5, 8, 11, 14 are DINOv2 x {dfu_healthy, dfu_nondfu, all, random_mixed}; index 2 is
DINOv2 x dfu_only). The skip-guard keeps every completed cell:

```bash
sbatch --array=5,8,11,14 slurm/run_composition_matrix_aicr.sh
```

**A8. Bring the results back (AICR has no git push auth).** Relay through an authenticated machine. From
your Mac, in the repo:

```bash
GIT_SSH_COMMAND="ssh -o BatchMode=yes" git fetch \
  "ssh://bandari_ru_neu@login.aicr.ai/home/bandari_ru_neu/DiaFoot.AI" main
ORIG=$(gh api user -q .login)
gh auth switch --user Ruthvik-Bandari
git -c credential.helper= -c credential.helper="!$(command -v gh) auth git-credential" \
    push origin FETCH_HEAD:main
gh auth switch --user "$ORIG"          # restore your default account
```

`results/` is gitignored, so force-add on the cluster before the relay:
`git add -f results/composition results/composition_comparison.* data/splits/cv/folds_manifest.json`.

### 9.5 Northeastern Explorer (H200)

Explorer works but is slower (one to two days) because of a per-user submitted-job QOS limit and offline
compute nodes. The committed `slurm/run_composition_matrix.sh` targets Explorer. Differences from AICR:

- Load modules and activate the venv: `source /etc/profile; module purge; module load cuda/12.8.0
  python/3.13.5; source .venv/bin/activate`.
- Compute nodes have **no internet**, so pre-cache pretrained weights on the login node and the script
  sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`:
  ```bash
  mkdir -p ~/.cache/torch/hub/checkpoints
  curl -L -o ~/.cache/torch/hub/checkpoints/mit_b0.pth \
    https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b0.pth
  curl -L -o ~/.cache/torch/hub/checkpoints/efficientnet-b4-6ed6700e.pth \
    https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
  # dinov2 is already cached from prior training; confirm all three build offline via srun.
  ```
- The array is throttled (`--array=0-14%4`) to respect the QOS limit. If even 15 tasks trip it, submit in
  halves: `sbatch --array=0-7%4 ...` then `sbatch --array=8-14%4 ...`.
- Run heavy audits and pre-warms via `srun` on a compute node, never on the login node (it OOM-kills
  heavy Python model builds).

Full Explorer copy-paste steps: [COMPOSITION_EXPERIMENT_RUNBOOK.md](COMPOSITION_EXPERIMENT_RUNBOOK.md).

### 9.6 Aggregate, verify, troubleshoot

Aggregate into the paper table (CPU, login node is fine):

```bash
python scripts/aggregate_composition_results.py \
    --results-dir results/composition \
    --output-json results/composition_comparison.json \
    --output-md   results/composition_comparison.md
cat results/composition_comparison.md
```

Verify:

- `ls results/composition/*.json | wc -l` returns **75**.
- The ordering in `composition_comparison.md` is DFU+Healthy > DFU-only > All > DFU+Non-DFU >
  Random-mixed on all three architectures.
- Spot-check provenance: `python -c "import json;
  print(json.load(open('results/composition/unetpp_dfu_only_seed42_fold0.json'))['provenance'])"` shows
  the test-split SHA above, checkpoint SHA, git commit, seed, fold, and per-class counts.

Troubleshooting:

| Symptom | Cause | Fix |
|---|---|---|
| `OSError [Errno 39] Directory not empty`, DINOv2 cells die in seconds | Concurrent `torch.hub.load` race | Pre-cache DINOv2 serially on login (A4), resubmit indices `5,8,11,14` |
| ~14 of 15 tasks fail together, NFS write hangs | `/home` quota blown by unpruned per-epoch checkpoints | Symlink `checkpoints/` to `/scratch` (A3); `quota -s` |
| Tasks stuck `PD (Priority)` with idle GPUs | 24h wall blocks backfill | Use 8h wall, or `scontrol update job <id> TimeLimit=08:00:00` (A6) |
| smp hangs for minutes per cell | Offline compute node without cached weights | Explorer: pre-cache + `HF_HUB_OFFLINE=1` (9.5). AICR: not needed |
| Login node OOM-kills your audit/pre-warm | Tight login-node memory | Run it via `srun` on a compute node |
| Cluster cannot `git push` | No git auth on the cluster | Relay through an authenticated machine (A8) |

### 9.7 Other SLURM jobs

`slurm/` also holds `train_single_gpu.sh`, `train_multi_gpu.sh` (DDP), `train_dinov2.sh`,
`train_ablation.sh` (12-config sweep), `run_ablation.sh`, `run_cross_val.sh`, `run_nnunet_train.sh`,
`evaluate.sh`, `evaluate_baseline_split.sh`, `rerun_metrics_h200.sh` (full metrics refresh), and
`run_fairness.sh`.

---

## 10. Composition-pipeline code reference

Complete technical reference for the code that produces the composition results.

Data flow:

```
data/splits/{train,val}.csv
        │  make_cv_folds.py  (patient-grouped k-fold, once)
        ▼
data/splits/cv/fold{0..4}/{train,val}.csv  +  folds_manifest.json
        │  run_composition_experiment.py  (one cell = arch × composition × fold)
        │     • composition.py filters/draws the training rows
        │     • Trainer fits; best ckpt picked by val/loss
        │     • run_segmentation_eval on the FIXED clean test.csv
        │     • composition_report.py summarizes + stamps provenance
        ▼
results/composition/{arch}_{comp}_seed42_fold{N}.json   (75 files)
        │  aggregate_composition_results.py
        ▼
results/composition_comparison.{json,md}
```

### 10.1 `src/data/composition.py`

Pure, image-free, unit-tested helpers. A *row* is one split-CSV record (`dict`); the only column reasoned
about is `class` in `{healthy, non_dfu, dfu}`. Constants: `POSITIVE_CLASS = "dfu"`,
`NEGATIVE_CLASSES = ("healthy", "non_dfu")`, `ALL_CLASSES = ("healthy", "non_dfu", "dfu")`.

```python
COMPOSITIONS = {
    "dfu_only":    ("dfu",),
    "dfu_healthy": ("dfu", "healthy"),
    "dfu_nondfu":  ("dfu", "non_dfu"),
    "all":         ("healthy", "non_dfu", "dfu"),
}   # random_mixed is NOT here: it is a size-matched draw, not a class subset
```

| Function | Signature | Behavior |
|---|---|---|
| `filter_by_classes` | `(rows, include) -> list[Row]` | Keep rows whose `class` is in `include` (order preserved). |
| `select_composition` | `(rows, composition) -> list[Row]` | Categorical preset from `COMPOSITIONS`. `ValueError` on unknown name. |
| `subsample_negatives` | `(rows, neg_frac, seed=42, sort_key="image") -> list[Row]` | Keep all DFU positives + a deterministic, class-stratified `neg_frac` of each negative class. `0.0` = DFU-only; `1.0` = all. |
| `random_mixed` | `(rows, seed=42, sort_key="image") -> list[Row]` | Draw `N` rows uniformly across all classes where `N` = number of DFU rows (size-matched control). Deterministic. |
| `group_kfold` | `(rows, n_folds=5, seed=42, group_key="patient_id") -> list[(train, val)]` | Grouped k-fold: rows sharing `group_key` stay in one fold. Falls back to `image`. |
| `class_counts` | `(rows) -> dict[str,int]` | `{class: count}`. |
| `read_split_csv` / `write_split_csv` | | CSV I/O helpers. |

Determinism: the driver uses `cell_seed = seed + fold`, so each fold gets a distinct training seed and a
distinct random-mixed / subsample draw.

### 10.2 `scripts/run_composition_experiment.py`

Trains one cell and writes one provenance-stamped JSON. Concurrency-safe: each cell writes its filtered
split CSVs to its own directory (`data/splits/_composition/<run_tag>/`). Selection is mutually exclusive:
exactly one of `--composition` or `--neg-frac` is required.

| Flag | Default | Meaning |
|---|---|---|
| `--arch` | (required) | `unetpp` \| `segformer` \| `dinov2`. |
| `--composition` | (one required) | `dfu_only` \| `dfu_healthy` \| `dfu_nondfu` \| `all` \| `random_mixed`. |
| `--neg-frac` | (one required) | Float in [0,1] for the negative-ratio sweep. |
| `--fold` | `None` | CV fold index; reads `{cv-dir}/fold{N}/{train,val}.csv`. |
| `--cv-dir` | `data/splits/cv` | Shared folds location. |
| `--seed` | `42` | Base seed (per-cell seed is `seed + fold`). |
| `--epochs` / `--batch-size` / `--num-workers` | `50` / `16` / `8` | Training loop. |
| `--lr` / `--weight-decay` | `1e-4` / `1e-2` | AdamW. |
| `--image-size` | arch default | 512 for unetpp/segformer, 518 for dinov2. |
| `--encoder-weights` | `imagenet` | `imagenet` or `none`. |
| `--save-qualitative` / `--n-qualitative` | off / `8` | Save predicted masks for fixed DFU test indices. |
| `--splits-dir` / `--work-dir` / `--results-dir` | `data/splits` / `checkpoints/composition` / `results/composition` | I/O roots. |
| `--device` / `--verbose` | `cuda` / off | |

Fixed recipe (identical across cells): model per `--arch` (`efficientnet-b4`+scSE; `mit_b0`;
`dinov2_vitb14` frozen + UPerNet), `AdamW`, `CosineAnnealingWithWarmup(warmup_epochs=5)`, `DiceCELoss`,
`TrainConfig(precision="bf16-mixed", gradient_clip=1.0, monitor_metric="val/loss", monitor_mode="min",
early_stopping_patience=15)`. Best checkpoint = lowest `val/loss` in the filename; evaluated on the full
clean `test.csv`.

### 10.3 `scripts/make_cv_folds.py`

`--splits-dir` (default `data/splits`), `--cv-dir` (`data/splits/cv`), `--n-folds` (`5`), `--seed`
(`42`), `--group-key` (`patient_id`, falls back to `image`). Writes `fold{0..4}/{train,val}.csv` and
`folds_manifest.json`.

### 10.4 `scripts/aggregate_composition_results.py`

`--results-dir` (`results/composition`), `--output-json`, `--output-md`. Reads the 75 per-cell JSONs,
writes the fold-averaged table and per-cell numbers with paired-bootstrap significance vs `dfu_only`.

### 10.5 `src/evaluation/composition_report.py`

`bootstrap_ci`, `paired_delta_ci`, `fp_rate_on_empty` (emptiness = ground-truth area 0, not the class
label, because non-DFU images carry real masks), `summarize_run`, `sha256_file`, `git_commit`,
`build_provenance`.

### 10.6 Result JSON schema

One file per cell: `results/composition/{arch}_{comp}_seed42_fold{N}.json`.

```
run_tag, arch, composition, seed, fold
summary
  dfu_only   aggregate{dice,iou,hd95,nsd_2mm,nsd_5mm,wound_area_mm2,wound_area_gt_mm2}
             -> each {mean,std,median,min,max}; dice_ci/iou_ci {mean,ci_low,ci_high,n}
  mixed      same structure, full mixed test set
  false_positive_on_empty  {n_empty, n_false_positive, fp_rate}
  per_image  {dfu_dice[], mixed_dice[], labels[]}
learning_curve  {"train/loss":[...], "val/loss":[...], "val/dice":[...]}   # per epoch
qualitative_dir, provenance{timestamp_utc, git_commit, split_csv, split_csv_sha256,
             checkpoint, checkpoint_sha256, versions, fold, cell_seed, image_size,
             lr, epochs, batch_size, n_train_by_class, n_val_by_class, n_test, train_csv}
```

Every cell's `provenance.split_csv_sha256` should equal the canonical
`d758d68928172c348075a478f7eaa1e496efbd2bf5d51772055126e6ac977851`.

### 10.7 SLURM array scripts and tests

- `slurm/run_composition_matrix.sh` (Explorer, committed): `--gres=gpu:h200:1 --array=0-14%4
  --time=08:00:00`, module loads, offline env.
- `slurm/run_composition_matrix_aicr.sh` (AICR B200, contents in 9.4-A5): `--partition=b200-batch
  --account=p2026_0017_neu --gres=gpu:b200:1 --array=0-14 --time=08:00:00`, no modules, no offline env.
- Index decode: `ARCH = ARCHS[idx % 3]`, `COMP = COMPOSITIONS[idx // 3]`; DINOv2 cells are indices 2, 5,
  8, 11, 14.
- Tests: `tests/test_composition.py`, `tests/test_composition_report.py`,
  `tests/test_aggregate_composition.py`.

---

## 11. Evaluation and audit suite

DiaFoot.AI ships a deep evaluation layer (`src/evaluation/`, `scripts/`):

- **Core metrics** (`metrics.py`): Dice, IoU, HD95, surface Dice (NSD), wound area in mm2.
- **Composition report** (`composition_report.py`): bootstrap CIs, paired deltas, FP-on-empty,
  provenance (Section 10.5).
- **Calibration** (`calibration.py`): Brier score, expected calibration error, temperature scaling,
  defer-threshold tuning.
- **External validation** (`external_validation.py`, `run_external_validation.py`): internal-vs-external
  performance drop with bootstrap CIs.
- **Fairness** (`fairness.py`, `run_fairness_audit.py`): ITA skin-tone-stratified metrics.
- **Robustness** (`robustness.py`, `run_tta_eval.py`): corruption tests and test-time augmentation.
- **Shortcut audit** (`shortcut_audit.py`, `run_shortcut_audit.py`): perturb border/background, measure
  prediction drift.
- **Subgroup analysis** (`subgroup_analysis.py`, `run_subgroup_audit.py`): metrics by ITA / source /
  wound size with CIs.
- **Failure atlas** (`failure_atlas.py`, `run_failure_atlas.py`): rank and categorize the worst
  predictions with overlays.
- **Uncertainty** (`uncertainty.py`): MC-dropout, ensembling, conformal wound-area intervals.
- **Annotator agreement** (`annotator_agreement.py`): pairwise Dice, majority vote, STAPLE, human
  ceiling, Fleiss kappa.
- **Area agreement** (`area_agreement.py`, `run_area_agreement.py`): model vs manual wound area.
- **Leakage audit** (`leakage_audit.py`, `run_leakage_audit.py`): the split integrity gate (Section 6).

---

## 12. Repository map

```
DiaFoot.AI/
  src/
    data/         dataset build, cleaning, dedup, leakage audit, ITA, composition, torch dataset
    models/       dinov2_segmenter (largest), dinov2_classifier, unetpp, fusegnet, attention, ...
    evaluation/   metrics, calibration, fairness, robustness, composition_report, audits, ...
    training/     trainer, multitask_trainer, losses (Dice/CE/FocalTversky/Boundary), schedulers, ema
    inference/    pipeline (classify -> gated segment -> area), onnx_export, tta
    deploy/       app.py (FastAPI), middleware (rate limit, size cap), schemas
  scripts/        37 CLI entrypoints (data pipeline, training, composition study, eval, audits, export)
  configs/        data / model / training / deploy / ablation YAMLs
  slurm/          12 cluster job scripts (training, composition matrix, eval, audits)
  tests/          39 test files + conftest (data, models, training, eval, inference, deploy, repro)
  docs/           this documentation set + runbooks + diagrams
  frontend/       Next.js web app (separate)
  data/           processed images + splits + metadata (gitignored large dirs)
  results/        metrics JSON, composition study outputs (gitignored; force-added)
  checkpoints/    model weights (gitignored)
```

Central / largest modules: `src/deploy/app.py`, `src/models/dinov2_segmenter.py`,
`src/data/cleaning.py`, `src/evaluation/calibration.py`, `src/data/ita_analysis.py`,
`src/inference/pipeline.py`.

Key CLI entrypoints by theme:

- **Data pipeline:** `run_data_pipeline.py` (orchestrator), `integrate_azh_data.py`,
  `collect_healthy_feet.py`, `collect_non_dfu.py`, `run_preprocessing.py`, `rebuild_splits_strict.py`,
  `make_cv_folds.py`, `run_leakage_audit.py`, `run_ita_analysis.py`.
- **Training:** `train.py`, `train_classifier.py`, `run_cross_val.py`, `run_ablation.py`.
- **Composition study:** `run_composition_experiment.py`, `aggregate_composition_results.py`.
- **Evaluation:** `evaluate.py`, `evaluate_all.py`, `run_external_validation.py`, `run_tta_eval.py`,
  `run_failure_atlas.py`, `summarize_cv_results.py`, `visualize_results.py`, `predict.py`.
- **Audits:** `run_fairness_audit.py`, `run_shortcut_audit.py`, `run_subgroup_audit.py`.
- **Export / repro:** `export_onnx.py`, `run_onnx_parity.py`, `run_repro_bundle.py`.

---

## 13. Configuration system

Configs are grouped YAMLs under `configs/`:

- **`data/`**: per-dataset registry (`fuseg`, `dfuc2022`, `healthy_feet`, `non_dfu`, `combined`) plus
  `cleaning.yaml` thresholds.
- **`model/`**: architecture specs (`unetpp_efficientnet`, `fusegnet`, `unetpp_multitask`, `nnunet_v2`,
  `classifier`).
- **`training/`**: full run configs (`baseline`, `advanced`, `multitask`, `dinov2_baseline`) with `seed`,
  `data`, `model`, `training` (optimizer/scheduler/loss/ema), `evaluation`, `logging`, `hardware`
  blocks, plus HPC overlays (`hpc.yaml`, `dinov2_hpc.yaml`).
- **`ablation/`**: 12 thin overrides on `training/baseline.yaml` for loss / encoder / data / attention.
- **`deploy/`**: `api.yaml` (host/port, size cap, rate limit, thresholds, checkpoints) and
  `onnx_export.yaml` (opset 17, input shape `[1,3,518,518]`).

Note: the composition study does not use the `configs/ablation/data_*.yaml` files; it selects rows
programmatically via `src/data/composition.py` for reproducibility and concurrency safety.

---

## 14. REST API

FastAPI service in `src/deploy/app.py` (title "DiaFoot.AI v2"). On startup it builds `InferencePipeline`
from the DINOv2 classifier and segmenter checkpoints. Middleware: CORS, a max-content-length cap on
`/predict`, and a per-minute rate limiter on `/predict`. Behavior is tuned by `DIAFOOT_*` env vars
(thresholds, checkpoint paths, backbone, device, prediction-log path).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, `model_loaded` flag, version. |
| GET | `/model/info` | Model names, input size `[518,518]`, class count, confidence/defer thresholds, size and rate limits. |
| POST | `/predict` | Upload an image, run the full pipeline; returns classification + probabilities, defer decision/reason, quality flags, `has_wound`, `wound_area_mm2`, `wound_coverage_pct`, base64 mask, diagnostics, timing. Optionally logs a `PredictionLogEvent` (JSONL) for drift monitoring. |

Run it: [howto-serve-api.md](howto-serve-api.md).

---

## 15. Reproduction

1. Install: see the README quickstart (`pip install -e .` or `requirements*.txt`).
2. Rebuild the dataset from sources: `scripts/download_data.sh` then `run_data_pipeline.py` (FUSeg + AZH
   are public; DFUC-2022 is license-gated and excluded; healthy/non-DFU from Mendeley and Kaggle).
3. Gate on clean splits: `run_leakage_audit.py` must report `has_any_leakage: false` (Section 9.2).
4. Train the deployed models ([howto-train.md](howto-train.md)) or run the composition study (Section 9).
5. Emit a reproducibility bundle: `run_repro_bundle.py` records git commit, environment, and package
   versions. Every composition result JSON additionally stamps the test-split SHA, checkpoint SHA, seed,
   fold, and git commit.

---

## 16. Deployment (Docker, ONNX)

- **Docker**: see `docker/` for the service image.
- **ONNX**: `export_onnx.py` exports the DINOv2 / U-Net++ models (opset 17, input `[1,3,518,518]`),
  validates PyTorch-to-ONNX parity, and benchmarks; `run_onnx_parity.py` checks parity on a split.
  Parity is about 99.99% and ONNX runs several times faster, the intended serving path.

---

## 17. Testing

39 test files under `tests/` (plus `conftest.py`) cover data pipeline and dataset construction, models,
training, the full evaluation and audit suite, inference and deploy, and reproducibility. The
composition study is covered by `test_composition.py`, `test_composition_report.py`, and
`test_aggregate_composition.py`. Run `pytest` (a non-GPU subset runs on CPU).

---

## 18. Limitations

- **Not a medical device.** Research code; no regulatory clearance.
- **Patient-level independence is not guaranteed.** The public sources lack real patient IDs, so folds
  and splits are grouped by the strongest available image-provenance identifier.
- **Triage classifier does not transfer.** It collapses on unseen sources (Section 4a). Segmentation
  transfers far better.
- **Non-DFU pool contamination.** The non-DFU set is ~16% diabetic, which makes the composition-study
  conclusion conservative (Section 8.7).
- **Skin-tone coverage is limited**, so fairness conclusions are preliminary.
- **Licensing constrains the data release.** FUSeg/AZH masks need permission to redistribute; Kaggle
  normals have no redistribution grant; only the Mendeley-derived masks ship freely.

---

## 19. The manuscript

The composition study is written up as an SPIE manuscript, *"Beyond Bigger Datasets: How Training Data
Composition Influences Diabetic Foot Ulcer Segmentation"* (with Prof. Mohammad Eslami, Harvard). The
source lives outside this repo (`manuscript/main.tex`, synced to Overleaf) and reports the same numbers
as Section 8, with the honest framing that what a training set contains matters more than how large it
is. A four-page version (`main_4pages.tex`) carries the acceptance decision; the long version is the
camera-ready.

---

## 20. Focused / shorter documents

This file is the complete reference. For quicker, task-scoped reading, these shorter documents also
exist (their detail is fully contained above):

| Document | For |
|---|---|
| [tutorial-getting-started.md](tutorial-getting-started.md) | Install to first prediction, from zero |
| [howto-run-data-pipeline.md](howto-run-data-pipeline.md) | Build the dataset from sources |
| [howto-train.md](howto-train.md) | Train the deployed models |
| [howto-serve-api.md](howto-serve-api.md) | Serve the REST API |
| [reference-cli.md](reference-cli.md) | Every CLI script and its flags |
| [reference-api.md](reference-api.md) | REST endpoints and schemas |
| [reference-architecture.md](reference-architecture.md) | `src/` package/module map |
| [explanation-pipeline-design.md](explanation-pipeline-design.md) | Why cascaded multi-task; the leakage story |
| [COMPOSITION_EXPERIMENT_RUNBOOK.md](COMPOSITION_EXPERIMENT_RUNBOOK.md) | Explorer copy-paste steps |
| [HPC_HONEST_RERUN_RUNBOOK.md](HPC_HONEST_RERUN_RUNBOOK.md) | Rebuilding clean splits after the leakage fix |
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | Full project report |

Architecture diagrams are in `docs/diagrams/` (`architecture`, `data-pipeline`, `inference-pipeline`, as
`.png`/`.svg`/`.mmd`/`.excalidraw`).
