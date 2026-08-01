#!/bin/bash
# DiaFoot.AI composition matrix on AICR/B200. 75 cells = 5 comps x 3 archs x 5 folds,
# as a 15-task array (comp x arch); each task runs its 5 folds sequentially.
# AICR: no submit cap, 32-GPU/user, 24h wall, compute nodes HAVE internet.
#SBATCH --job-name=diafoot-comp
#SBATCH --partition=b200-batch
#SBATCH --account=p2026_0017_neu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/slurm/%A_%a_composition.out
#SBATCH --error=logs/slurm/%A_%a_composition.err
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p logs/slurm results/composition
COMPOSITIONS=(dfu_only dfu_healthy dfu_nondfu all random_mixed)
ARCHS=(unetpp segformer dinov2)
IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCHS[$(( IDX % 3 ))]}
COMP=${COMPOSITIONS[$(( IDX / 3 ))]}
for FOLD in 0 1 2 3 4; do
  RESULT="results/composition/${ARCH}_${COMP}_seed42_fold${FOLD}.json"
  if [ -f "$RESULT" ]; then echo "[$(date)] task ${IDX} | skip existing: ${RESULT}"; continue; fi
  QUAL=""; [ "$FOLD" -eq 0 ] && QUAL="--save-qualitative --n-qualitative 8"
  echo "[$(date)] task ${IDX} | arch=${ARCH} comp=${COMP} fold=${FOLD}"
  # shellcheck disable=SC2086
  .venv/bin/python scripts/run_composition_experiment.py \
      --arch "${ARCH}" --composition "${COMP}" --fold "${FOLD}" \
      --seed 42 --device cuda --epochs 50 --batch-size 16 --num-workers 16 $QUAL
done
echo "[$(date)] finished task ${IDX} (arch=${ARCH} comp=${COMP}, folds 0-4)"
