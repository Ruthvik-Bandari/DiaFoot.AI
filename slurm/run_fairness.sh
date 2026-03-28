#!/bin/bash
#SBATCH --job-name=diafoot-fairness
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_fairness.out
#SBATCH --error=logs/%j_fairness.err

module load cuda/12.8.0 python/3.13.5
source ~/DiaFoot.AI-v2/.venv/bin/activate
cd ~/DiaFoot.AI-v2

python scripts/run_fairness_audit.py \
    --cls-checkpoint checkpoints/dinov2_classifier/best.pt \
    --seg-checkpoint checkpoints/dinov2_segmenter/best.pt \
    --device cuda
