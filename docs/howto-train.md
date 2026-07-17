# How to train and evaluate a model

This guide shows you how to train a DiaFoot.AI classifier or segmenter from prepared
splits, run it on SLURM, evaluate it against the held-out test set, and confirm the run
produced real checkpoints and metrics.

## Prerequisites

Before you train, make sure the following are in place.

- **The package is installed in editable mode with dev extras.** From the repository root:

  ```bash
  pip install -e ".[dev]"
  ```

  This installs the `diafootai` package plus the `diafootai-train` and `diafootai-eval`
  console scripts.

- **A GPU for training.** Training uses `bf16-mixed` precision, so you want a CUDA GPU that
  supports BF16 (for example an H100 or A100). Evaluation can fall back to CPU, and so can
  training if you pass `--device cpu`, but a full run on CPU is slow.

- **Prepared splits in `data/splits`.** The training and evaluation scripts read
  `data/splits/train.csv`, `data/splits/val.csv`, and `data/splits/test.csv`. If these do not
  exist yet, build them first with the data pipeline. See
  [howto-run-data-pipeline.md](howto-run-data-pipeline.md).

- **A chosen config.** The training configs live in `configs/training/`. The default is
  `configs/training/dinov2_baseline.yaml`. You can also use `configs/training/multitask.yaml`,
  `configs/training/dinov2_hpc.yaml`, or an ablation config under `configs/ablation/`.

## Train a model

Training runs through `scripts/train.py`. The `--task` flag is required and picks one of three
modes: `classify` (DINOv2 3-class triage), `segment` (DINOv2 wound segmenter), or
`segment-unetpp` (the legacy U-Net++ baseline kept for ablation comparison).

Train the DINOv2 triage classifier on a single local GPU:

```bash
python scripts/train.py \
    --task classify \
    --config configs/training/dinov2_baseline.yaml \
    --splits-dir data/splits \
    --backbone dinov2_vitb14 \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --weight-decay 1e-2
```

Train the DINOv2 wound segmenter (this mode automatically filters the data to DFU images
only, which is the ablation-backed default):

```bash
python scripts/train.py \
    --task segment \
    --config configs/training/dinov2_baseline.yaml \
    --backbone dinov2_vitb14 \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --use-lora
```

You can now add LoRA adapters with `--use-lora` (tuned by `--lora-rank`, default 8, and
`--lora-alpha`, default 16), or unfreeze the backbone for full fine-tuning with
`--unfreeze-backbone`. The console script `diafootai-train` is equivalent to
`python scripts/train.py`.

**Choosing a config.** The baseline (`dinov2_baseline.yaml`) trains a single DINOv2 head and is
the right starting point. The multitask config (`multitask.yaml`) drives the combined
classify plus segment plus Wagner-stage U-Net++ recipe. The ablation configs
(`configs/ablation/*`) swap one knob at a time to answer a specific question (see
[Run ablations](#run-ablations)).

### Key config knobs

`scripts/train.py` takes the core knobs from the command line (`--epochs`, `--batch-size`,
`--lr`, `--weight-decay`, `--backbone`, and the LoRA flags). The YAML configs record the full
training recipe that the HPC and multitask entry points consume. The knobs worth knowing:

- **epochs.** `dinov2_baseline.yaml` documents a three-phase schedule (20-epoch linear probe,
  50-epoch LoRA fine-tune, 20-epoch partial unfreeze). `multitask.yaml` sets `epochs: 100`.
- **batch-size.** 16 for the ViT baseline, 24 for the multitask U-Net++. Lower this first if
  you hit out-of-memory errors.
- **lr.** `1e-4` by default. The baseline phases step down from `1e-3` (head only) to `5e-5`
  (LoRA) to `1e-5` (backbone).
- **loss.** The baseline uses `dice_ce` for segmentation and `focal` (gamma 2.0) for
  classification. The multitask config uses `dice_boundary`, `focal`, and
  `label_smoothing_ce` across its three heads.
- **ema.** `multitask.yaml` enables an exponential moving average of the weights with
  `decay: 0.999` for a more stable final model.
- **scheduler warmup.** Both configs use cosine annealing. The baseline warms up over 5 epochs
  and the multitask config over 10 (`scripts/train.py` uses a 5-epoch warmup).

## Train on SLURM / HPC

For cluster runs, submit the batch script instead of calling Python directly:

```bash
sbatch slurm/train_single_gpu.sh
```

This requests one H100, loads CUDA and Python, activates `.venv`, verifies BF16 support, and
then runs `scripts/train.py`. The script takes two positional arguments: the task (default
`segment`) and the config path (default `configs/training/dinov2_baseline.yaml`). You can also
set the backbone through the `BACKBONE` environment variable. For example:

```bash
BACKBONE=dinov2_vitl14 sbatch slurm/train_single_gpu.sh classify configs/training/dinov2_hpc.yaml
```

For distributed runs use `slurm/train_multi_gpu.sh`, and for cluster ablation sweeps use
`slurm/train_ablation.sh` or `slurm/run_ablation.sh`. Cross-validation has its own job at
`slurm/run_cross_val.sh`.

## Evaluate

Evaluation runs through `scripts/evaluate.py`. Both `--task` and `--checkpoint` are required.
Evaluation always uses the test split at `data/splits/test.csv` and falls back to CPU
automatically when CUDA is not available.

Evaluate a trained classifier:

```bash
python scripts/evaluate.py \
    --task classify \
    --checkpoint checkpoints/dinov2_classifier/best.pt \
    --splits-dir data/splits \
    --backbone dinov2_vitb14 \
    --device cuda
```

Evaluate a trained DINOv2 segmenter:

```bash
python scripts/evaluate.py \
    --task segment \
    --checkpoint checkpoints/dinov2_segmenter/best.pt \
    --device cuda
```

To evaluate the legacy U-Net++ baseline, add `--model unetpp`:

```bash
python scripts/evaluate.py \
    --task segment \
    --model unetpp \
    --checkpoint checkpoints/unetpp_baseline/best.pt
```

The console script `diafootai-eval` is equivalent to `python scripts/evaluate.py`.

For a full comparison across every trained segmentation checkpoint (U-Net++, FUSegNet, DINOv2,
and the ablation variants), run:

```bash
python scripts/evaluate_all.py --device cuda
```

For a robust cross-validated estimate, run each fold of the 5-fold cross-validation and then
summarize:

```bash
python scripts/run_cross_val.py --fold 0 --device cuda --epochs 50
python scripts/summarize_cv_results.py
```

Run folds 0 through 4 (as a SLURM array job on a cluster) to cover all five.

For the reported, leakage-audited results, see the main [README](../README.md).

## Run ablations

The data composition ablation trains three segmenters that differ only in which classes they
see. Run each variant:

```bash
python scripts/run_ablation.py --variant dfu_only --device cuda --epochs 50
python scripts/run_ablation.py --variant dfu_nondfu --device cuda --epochs 50
python scripts/run_ablation.py --variant all --device cuda --epochs 50
```

Each variant writes to its own checkpoint directory (for example
`checkpoints/ablation_dfu_only`). To ablate other axes, use the configs under
`configs/ablation/`: `loss_*` (unified focal, focal Tversky, Dice CE, Dice boundary),
`encoder_*` (EfficientNet-B4, EfficientNet-B7, ConvNeXt, ViT), `data_*` (DFU only, DFU plus
healthy, DFU plus healthy plus non-DFU), and `attention_pscse.yaml`.

## Verification

Confirm the run actually worked by checking three things.

- **Checkpoints were written.** Each task saves to its own directory under `checkpoints/`, for
  example `checkpoints/dinov2_classifier/` for the classifier and
  `checkpoints/dinov2_segmenter/` for the segmenter. Confirm a checkpoint file exists:

  ```bash
  ls -la checkpoints/dinov2_classifier/
  ```

- **Metrics were logged.** Training logs to Weights & Biases under the project
  `DiaFootAI-v2`. Open that project to see loss and metric curves. If you do not have a wandb
  account, watch the console log, which reports per-epoch train and validation numbers.

- **Evaluation produced result files.** `scripts/evaluate.py` writes to `results/`. A classifier
  run creates `results/classification_metrics.json` and
  `results/classification_calibration.json`. A segmenter run creates
  `results/segmentation_metrics.json`. `evaluate_all.py` writes
  `results/all_models_comparison.json`, and each cross-validation fold writes
  `results/cv_fold<n>.json`. Confirm and inspect one:

  ```bash
  ls -la results/
  cat results/segmentation_metrics.json
  ```

## Troubleshooting

- **CUDA out of memory.** Lower `--batch-size` (for example from 16 to 8 or 4). If you enabled
  `--unfreeze-backbone`, drop it or switch to a smaller backbone such as `dinov2_vits14`.
- **Missing splits.** If the run fails to find `data/splits/train.csv` or `test.csv`, the
  splits have not been built. Run the data pipeline first, described in
  [howto-run-data-pipeline.md](howto-run-data-pipeline.md).
- **Wrong device or no GPU.** `scripts/train.py` uses `--device cuda` by default and does not
  fall back, so on a machine without a GPU pass `--device cpu` explicitly.
  `scripts/evaluate.py` falls back to CPU on its own when CUDA is unavailable.
- **No checkpoint to evaluate.** `--checkpoint` is required and the file must exist. Train the
  matching task first, then point `--checkpoint` at the produced `best.pt`.

## Related

- [howto-run-data-pipeline.md](howto-run-data-pipeline.md) — build the splits this guide needs.
- [reference-cli.md](reference-cli.md) — full flag reference for every script.
- [explanation-pipeline-design.md](explanation-pipeline-design.md) — why the pipeline is
  designed this way.
- [README](../README.md) — project overview and reported results.
