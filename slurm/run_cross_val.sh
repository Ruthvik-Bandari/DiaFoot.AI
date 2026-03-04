#!/bin/bash
#SBATCH --job-name=diafoot-cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%A_%a_cv.out
#SBATCH --error=logs/%A_%a_cv.err

module load cuda/12.8.0 python/3.13.5
source ~/DiaFoot.AI-v2/.venv/bin/activate
cd ~/DiaFoot.AI-v2

echo "Running fold $SLURM_ARRAY_TASK_ID / 5"

python scripts/run_cross_val.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 8 \
    --verbose
