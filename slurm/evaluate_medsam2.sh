#!/bin/bash
# DiaFoot.AI v2 — MedSAM2 Evaluation
# Usage: sbatch slurm/evaluate_medsam2.sh [checkpoint]

#SBATCH --job-name=medsam2-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j_medsam2_eval.out
#SBATCH --error=logs/slurm/%j_medsam2_eval.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

CKPT=${1:-checkpoints/medsam2_lora/best_epoch015_0.8151.pt}
SPLIT=${2:-val}
INCLUDE_CLASSES=${3:-dfu,non_dfu}

mkdir -p logs/slurm results

echo "Evaluating MedSAM2 checkpoint=${CKPT} split=${SPLIT} classes=${INCLUDE_CLASSES} | $(date)"

python scripts/evaluate_medsam2.py \
  --checkpoint "$CKPT" \
  --splits-dir data/splits \
  --split "$SPLIT" \
  --include-classes "$INCLUDE_CLASSES" \
  --device cuda \
  --out-json results/segmentation_metrics_medsam2.json

echo "Finished: $(date)"
