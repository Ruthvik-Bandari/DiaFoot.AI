#!/bin/bash
# DiaFoot.AI v2 — Baseline Segmentation Eval (matched split/classes)
# Usage: sbatch slurm/evaluate_baseline_split.sh [checkpoint] [split] [classes]

#SBATCH --job-name=baseline-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j_baseline_eval.out
#SBATCH --error=logs/slurm/%j_baseline_eval.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

CKPT=${1:-checkpoints/cv_fold3/best_epoch019_0.1301.pt}
SPLIT=${2:-val}
CLASSES=${3:-dfu,non_dfu}

mkdir -p logs/slurm results

echo "Evaluating baseline checkpoint=${CKPT} split=${SPLIT} classes=${CLASSES} | $(date)"

python scripts/evaluate_baseline_split.py \
  --checkpoint "$CKPT" \
  --splits-dir data/splits \
  --split "$SPLIT" \
  --include-classes "$CLASSES" \
  --device cuda \
  --out-json results/segmentation_metrics_baseline_matched.json

echo "Finished: $(date)"
