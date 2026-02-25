#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DiaFoot.AI v2 — Multi-GPU DDP Training (4x H100)
# Usage: sbatch slurm/train_multi_gpu.sh [config_path]
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=diafootai-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bandari.r@northeastern.edu

set -euo pipefail
module purge && module load cuda/12.4 python/3.12
source .venv/bin/activate

echo "Job ${SLURM_JOB_ID} | Node $(hostname) | GPUs ${CUDA_VISIBLE_DEVICES:-?} | $(date)"

# Verify all 4 GPUs
python -c "
import torch
n = torch.cuda.device_count()
assert n >= 4, f'Expected 4 GPUs, got {n}'
for i in range(n): print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

export TORCH_COMPILE_BACKEND=inductor
export TORCHINDUCTOR_MODE=reduce-overhead
export NCCL_DEBUG=INFO
mkdir -p logs/slurm

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train.py \
    --config "${1:-configs/training/advanced.yaml}" \
    --distributed ddp \
    --precision bf16-mixed \
    --compile true \
    --seed 42

echo "Finished: $(date)"
