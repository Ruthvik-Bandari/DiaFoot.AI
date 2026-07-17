# How to run the data pipeline and build leak-free splits

This guide walks you through turning collected raw images into a training-ready
dataset. When you finish, you will have a cleaned, preprocessed, stratified, and
leakage-audited dataset with `train.csv`, `val.csv`, and `test.csv` in
`data/splits/`, plus supporting reports in `data/metadata/`.

Leak-free splits are the point of this pipeline. If the same wound (or an
augmented or re-encoded copy of it) lands in both train and test, evaluation
numbers inflate and mean nothing. Every step below exists to prevent that, and
the final leakage audit proves it worked. For the story behind why this matters
and what the honest numbers look like, see
[explanation-pipeline-design.md](explanation-pipeline-design.md) and the results
tables in [../README.md](../README.md).

## Prerequisites

- **The package is installed.** From the project root, run `pip install -e .`
  (or `pip install -e ".[dev]"`). This installs `diafootai` v2.0.0 and its
  dependencies, and lets `scripts/*.py` import from `src/`.
- **Raw datasets are collected.** The pipeline reads from `data/raw/`. Populate
  it first with the collection scripts:
  - `python scripts/collect_healthy_feet.py` writes healthy-foot images to
    `data/raw/healthy/`.
  - `python scripts/collect_non_dfu.py` writes non-DFU wound images to
    `data/raw/non_dfu/`.
  - `python scripts/integrate_azh_data.py` pulls the AZH GitHub DFU set into
    `data/raw/azh_github/`.
- **You expect the 3-category dataset.** DiaFoot.AI trains on three classes:
  DFU wounds, healthy feet, and non-DFU conditions, roughly 8,105 images total
  (about 2,119 DFU, 3,300 healthy, and 2,686 non-DFU). The class definitions and
  mask conventions live in
  [../data/metadata/dataset_card.md](../data/metadata/dataset_card.md).

Run every command below from the project root so the default relative paths
(`data/raw`, `data/processed`, `data/splits`, `data/metadata`) resolve.

## The pipeline at a glance

The stages run in this order:

1. **Collect** raw images into `data/raw/` (`collect_healthy_feet.py`,
   `collect_non_dfu.py`, `integrate_azh_data.py`).
2. **Clean and dedup**: integrity checks, mask validation, and perceptual-hash
   duplicate detection (`run_cleaning.py`).
3. **Label and quality**: Wagner grade labels and per-image quality flags
   (`run_label_audit.py`, Wagner labeling).
4. **ITA skin tone**: compute the Individual Typology Angle so splits can be
   stratified by skin tone (`run_ita_analysis.py`).
5. **Preprocess**: resize to 512, pad, CLAHE, and binarize masks
   (`run_preprocessing.py`).
6. **Stratified splits**: build group-preserving 70/15/15 train/val/test splits
   (`run_data_pipeline.py`, then `rebuild_splits_strict.py`).
7. **Leakage audit**: verify zero overlap on all axes
   (`run_leakage_audit.py`).

```
 collect ──▶ clean/dedup ──▶ label/quality ──▶ ITA skin tone
                                                     │
                                                     ▼
 leakage audit ◀── strict splits (70/15/15) ◀── preprocess
```

You can run the whole thing with one command, or run any stage on its own. Both
paths are covered below.

## Run the whole pipeline

`run_data_pipeline.py` orchestrates the end-to-end sequence: it integrates the
AZH GitHub data into `data/processed/dfu/`, validates every processed
image-mask pair, generates group-preserving 70/15/15 stratified splits, runs a
leakage check, and writes a summary report.

```
python scripts/run_data_pipeline.py --verbose
```

The script accepts only three flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--data-dir` | `data` | Root that holds `raw/`, `processed/`, `splits/`, `metadata/`. |
| `--seed` | `42` | Seed for the split shuffle, so runs are reproducible. |
| `--verbose` | off | Switch logging from INFO to DEBUG. |

There is no `--skip-dedup` flag despite what an older usage string suggests, so
do not pass it.

What it writes:

- `data/processed/dfu/images/` and `data/processed/dfu/masks/`: AZH images,
  resized to 512 with padding, CLAHE-enhanced, and binarized masks.
- `data/splits/train.csv`, `val.csv`, `test.csv`: the stratified splits. Each row
  carries `image`, `mask`, `class`, `ita`, `ita_group`, `source_id`, and
  `patient_id`.
- `data/metadata/data_pipeline_report.json`: AZH integration counts, validation
  per class, split sizes and class distribution, patient and source overlap
  checks, and the leakage report.

The stratifier groups samples by `class + ita_group + source_id` and assigns
whole patient groups to a single split, so no patient straddles train and test.
The console prints the per-split class counts and the leakage summary at the end.
If the final leakage check flags near-duplicates, run the strict rebuild in the
next section, then re-audit.

## Run individual stages

Use these when you want to re-run one step, tune a threshold, or debug a stage in
isolation.

### Clean and dedup

`run_cleaning.py` runs the CleanVision and Cleanlab quality audit: integrity
checks, mask validation, and perceptual-hash duplicate detection.

```
# Audit every dataset under data/raw/dfu
python scripts/run_cleaning.py

# Audit one directory with a name for the report
python scripts/run_cleaning.py --data-dir data/raw/dfu/fuseg --name fuseg
```

Flags: `--data-dir` (default: audit all under `data/raw/dfu`), `--name`,
`--config` (default `configs/data/cleaning.yaml`), `--output-dir` (default
`data/metadata`), and `--verbose`.

It writes `data/metadata/quality_report_<name>.json` per dataset (for example
`quality_report_fuseg.json`), or `quality_report_*.json` for each dataset when
you audit all of them.

### ITA skin tone analysis

`run_ita_analysis.py` computes the Individual Typology Angle for every image and
bins each into a skin-tone category, so the splitter can stratify by skin tone.

```
python scripts/run_ita_analysis.py --verbose
```

Flags: `--data-root` (default `data/raw`), `--output-dir` (default
`data/metadata`), and `--verbose`.

It writes `data/metadata/ita_scores.csv` (per-image ITA values), per-dataset
files such as `ita_scores_fuseg.csv`, and a `data/metadata/ita_report.json`
summary with per-dataset mean, median, and category distribution.

### Preprocess

`run_preprocessing.py` resizes images to 512 with padding, applies CLAHE,
binarizes masks, and writes the results to `data/processed/`. By default it then
builds stratified splits.

```
# Preprocess and build splits
python scripts/run_preprocessing.py --verbose

# Preprocess only, skip split creation
python scripts/run_preprocessing.py --skip-splits
```

Flags: `--data-root` (default `data/raw`), `--output-dir` (default
`data/processed`), `--splits-dir` (default `data/splits`), `--target-size`
(default `512`), `--no-clahe`, `--skip-splits`, and `--verbose`.

It writes preprocessed `images/` and `masks/` under
`data/processed/<category>/`, and (unless you pass `--skip-splits`)
`data/splits/train.csv`, `val.csv`, `test.csv` plus
`data/splits/split_stats.json` with class and ITA distributions.

### Rebuild strict, leak-free splits

`rebuild_splits_strict.py` is the belt-and-suspenders step. It reads the existing
`train.csv`, `val.csv`, and `test.csv`, recomputes a robust source and patient
identity from each filename, groups near-duplicate images into components with a
class-aware dHash graph, and assigns whole components to one split. Because a
component (all patches, augmentations, and near-duplicates of one capture) never
gets broken apart, no duplicate can straddle two splits.

```
python scripts/rebuild_splits_strict.py --split-dir data/splits --near-threshold 6
```

Flags: `--split-dir` (default `data/splits`, note the singular name here),
`--seed` (default `42`), `--train-ratio` (default `0.70`), `--val-ratio`
(default `0.15`), and `--near-threshold` (default `6`, the maximum dHash Hamming
distance treated as a near-duplicate).

It overwrites `data/splits/train.csv`, `val.csv`, and `test.csv` in place with
the rebuilt, leak-free splits and prints a per-split class-count summary.

### Leakage audit

`run_leakage_audit.py` is the proof step. It loads the three split CSVs and
checks for overlap on four axes: exact path, canonical sample id (filename with
augmentation suffixes stripped), content hash, and perceptual near-duplicate.

```
python scripts/run_leakage_audit.py --verbose
```

Flags: `--splits-dir` (default `data/splits`, plural here), `--output` (default
`data/metadata/leakage_report.json`), `--near-threshold` (default `6`),
`--max-examples` (default `50`), and `--verbose`.

It writes the report to `--output` and prints the per-axis overlap counts and a
single `Any leakage signal: True/False` line. The script raises
`FileNotFoundError` if any of the three split CSVs is missing.

## Verification

To confirm the splits are leak-free, write a fresh audit report and read its
verdict.

Write a recheck report:

```
python scripts/run_leakage_audit.py \
    --splits-dir data/splits \
    --output data/metadata/leakage_report_recheck.json
```

Confirm the verdict is `False`:

```
python -c "import json; print(json.load(open('data/metadata/leakage_report_recheck.json'))['has_any_leakage'])"
```

A leak-free dataset prints `False`. In the committed
`data/metadata/leakage_report_recheck.json`, `has_any_leakage` is `false` and
every entry in `path_overlap`, `canonical_overlap`, `content_overlap`, and
`near_duplicates.counts` is `0` across `train_x_val`, `train_x_test`, and
`val_x_test`. If any axis is nonzero, run `rebuild_splits_strict.py` and audit
again.

Check the split sizes:

```
wc -l data/splits/train.csv data/splits/val.csv data/splits/test.csv
```

Each file has one header row, so subtract 1 from each line count. The committed
leak-free splits are 5,782 train, 1,162 val, and 1,161 test (8,105 total), which
match the counts in [../README.md](../README.md).

## Troubleshooting

**A stage reports missing raw data.** `run_preprocessing.py` logs
`Skipping <category>: <dir> not found` and moves on, and `run_data_pipeline.py`
silently skips AZH integration when `data/raw/azh_github/` is absent, so a run
can finish with far fewer samples than expected. Confirm you ran the three
collection scripts (`collect_healthy_feet.py`, `collect_non_dfu.py`,
`integrate_azh_data.py`) and that `data/raw/` has the expected `healthy/`,
`non_dfu/`, and `azh_github/` layout. If `run_leakage_audit.py` raises
`FileNotFoundError`, one of the split CSVs does not exist yet, so build the
splits before auditing.

**Dedup removes or merges too much.** The `--near-threshold` in
`rebuild_splits_strict.py` and `run_leakage_audit.py` is the maximum dHash
Hamming distance treated as a near-duplicate. A higher threshold merges more
images into each component (more aggressive), and a lower threshold merges fewer.
If distinct-but-similar images collapse into one component, lower
`--near-threshold` below the default of 6 and rebuild.

**A split is missing a class or looks imbalanced.** The splitter keeps whole
patient and near-duplicate groups together, so a class with very few groups can
run out of components before it can seed all three splits.
`rebuild_splits_strict.py` logs a warning like
`class '<name>': too few components to give split '<split>' its own` and leaves
that class out of a split rather than break a component (which would reintroduce
leakage). Inspect the per-split class counts in
`data/metadata/data_pipeline_report.json` or the printed summary. If a class is
too thin, collect more of it, or adjust `--train-ratio` and `--val-ratio`, then
rebuild and re-audit.

## Related

- [howto-train.md](howto-train.md) — train and evaluate on these splits.
- [explanation-pipeline-design.md](explanation-pipeline-design.md) — why the
  cascade exists and the full data-leakage story.
- [reference-cli.md](reference-cli.md) — every CLI script and its flags.
- [../README.md](../README.md) — honest results on these leak-free splits.
