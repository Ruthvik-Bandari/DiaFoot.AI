#!/bin/bash
#SBATCH --job-name=diafoot-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%A_%a_ablation.out
#SBATCH --error=logs/%A_%a_ablation.err

module load cuda/12.8.0 python/3.13.5
source ~/DiaFoot.AI-v2/.venv/bin/activate
cd ~/DiaFoot.AI-v2

VARIANTS=("dfu_only" "dfu_nondfu" "all")
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Running ablation: $VARIANT"

python scripts/run_ablation.py \
    --variant $VARIANT \
    --device cuda \
    --epochs 50 \
    --batch-size 16 \
    --num-workers 8 \
    --verbose
