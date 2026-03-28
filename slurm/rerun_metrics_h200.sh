#!/bin/bash
# DiaFoot.AI v2 — One-shot metrics rerun on H200
# Usage: sbatch slurm/rerun_metrics_h200.sh
#SBATCH --job-name=diafoot-metrics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j_metrics.out
#SBATCH --error=logs/slurm/%j_metrics.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.8.0 python/3.12
source .venv/bin/activate

export PYTHONPATH="$PROJECT_ROOT"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p logs/slurm results

CLASSIFIER_CKPT="checkpoints/dinov2_classifier/best.pt"
SEGMENTER_CKPT="checkpoints/dinov2_segmenter/best.pt"

echo "[1/6] leakage audit"
"$PROJECT_ROOT/.venv/bin/python" scripts/run_leakage_audit.py

echo "[2/6] internal classify eval"
"$PROJECT_ROOT/.venv/bin/python" scripts/evaluate.py --task classify --checkpoint "$CLASSIFIER_CKPT" --device cuda

echo "[3/6] internal segment eval"
"$PROJECT_ROOT/.venv/bin/python" scripts/evaluate.py --task segment --checkpoint "$SEGMENTER_CKPT" --device cuda

echo "[4/6] external classification benchmark"
"$PROJECT_ROOT/.venv/bin/python" scripts/run_external_validation.py \
  --external-split data/splits/external.csv \
  --cls-checkpoint "$CLASSIFIER_CKPT" \
  --output results/external_validation.json

echo "[5/6] external segmentation benchmark"
"$PROJECT_ROOT/.venv/bin/python" scripts/run_external_validation.py \
  --external-split data/splits/external_segmentation.csv \
  --seg-checkpoint "$SEGMENTER_CKPT" \
  --output results/external_segmentation_validation.json

echo "[6/6] reproducibility bundle"
"$PROJECT_ROOT/.venv/bin/python" scripts/run_repro_bundle.py

echo "Done: $(date)"
