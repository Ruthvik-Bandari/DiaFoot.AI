#!/bin/bash
#SBATCH --job-name=diafoot-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/slurm/%A_%a_cv.out
#SBATCH --error=logs/slurm/%A_%a_cv.err

set -euo pipefail

# Resolve project root from sbatch submit location (fallback to script parent).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"

module purge
module load cuda/12.8.0 python/3.12
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"
mkdir -p logs/slurm results

echo "Running fold $SLURM_ARRAY_TASK_ID / 5"

"$PROJECT_ROOT/.venv/bin/python" scripts/run_cross_val.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 8 \
    --verbose
