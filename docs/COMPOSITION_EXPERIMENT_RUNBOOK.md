# Composition Study — Explorer Runbook

Run the training-data-composition experiment on Northeastern Explorer, where the
images and GPUs live. Produces the paper's core result: with **architecture and
hyperparameters fixed**, how does the *composition* of the training set affect
diabetic-foot-ulcer segmentation? Every cell is evaluated on the **same fixed
clean test set**, with DFU-Dice / IoU / HD95 / NSD, 5-fold cross-validation
(mean ± std), and paired significance vs DFU-only.

Every step is gated by the one before it. **Do not `sbatch` until the clean-split
gate (step 3) passes** — training on leaky splits reproduces the exact bug this
study exists to avoid.

---

## What runs (75 cells = 5 × 3 × 5)

| Axis | Values |
|---|---|
| Composition | `dfu_only` · `dfu_healthy` · `dfu_nondfu` · `all` · `random_mixed` (size-matched to DFU-only) |
| Architecture | U-Net++ (EffB4-scSE) · SegFormer-B0 (MiT-B0) · DINOv2 ViT-B/14 + UPerNet |
| CV fold | 0–4 (patient-grouped, shared across all cells) |

`slurm/run_composition_matrix.sh` is an array of 75 cells. Each writes its own
filtered splits + checkpoint and emits `results/composition/<run_tag>.json`
(metrics + 95% CIs + false-positive-on-empty + learning curve + provenance).
Qualitative prediction masks are saved on fold 0 only (for the comparison figure).

---

## 0. Prereqs

- The composition-experiment code is on the branch Explorer will pull (the new
  `scripts/run_composition_experiment.py`, `scripts/make_cv_folds.py`,
  `scripts/aggregate_composition_results.py`, `src/data/composition.py`,
  `src/evaluation/composition_report.py`, `slurm/run_composition_matrix.sh`).
  **This requires the commit to be pushed to GitHub first** (you curate `main`).
- Clean, leak-free splits + images already on Explorer from the honest re-run:
  `data/splits/{train,val,test}.csv` and `data/processed/{dfu,healthy,non_dfu}/{images,masks}/`.
- The DINOv2 backbone is in the torch hub cache from the prior training (compute
  nodes may lack internet). SegFormer/U-Net++ ImageNet weights download from the
  login node on first use — pre-warm there if compute nodes are offline (see Notes).

Set your repo location once (adjust to wherever the repo lives on Explorer):

```bash
export DIAFOOT=$HOME/DiaFoot.AI-v2      # or /scratch/$USER/DiaFoot.AI
cd "$DIAFOOT"
```

## 1. Get the code onto Explorer

```bash
git fetch origin && git checkout main && git pull --ff-only
git log --oneline -3
ls scripts/run_composition_experiment.py scripts/make_cv_folds.py slurm/run_composition_matrix.sh
```

## 2. Load the environment (same modules/venv as the honest re-run)

```bash
source /etc/profile
module purge && module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate
export PYTHONPATH="$DIAFOOT"
python -c "import torch, segmentation_models_pytorch as smp; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'smp', smp.__version__)"
```

## 3. GATE — prove the splits are the clean set (do NOT skip)

```bash
python scripts/run_leakage_audit.py --splits-dir data/splits \
    --output data/metadata/leakage_report_composition.json --verbose
python - <<'PY'
import json
r = json.load(open("data/metadata/leakage_report_composition.json"))
flag = r.get("has_any_leakage", r.get("leakage", {}).get("has_any_leakage"))
assert flag is False, f"LEAKAGE PRESENT ({flag}) — STOP, rebuild clean splits first"
print("clean-split gate PASSED: has_any_leakage =", flag)
PY
```

If this fails, stop and rebuild clean splits (see `docs/HPC_HONEST_RERUN_RUNBOOK.md`
step 2) before continuing.

## 4. Generate the shared CV folds (once)

```bash
python scripts/make_cv_folds.py --n-folds 5 --seed 42
ls data/splits/cv/fold*/train.csv          # expect 5 folds
cat data/splits/cv/folds_manifest.json | python -m json.tool | head -30
```

These patient-grouped folds partition the train+val pool and are shared by every
cell, so all compositions/architectures see the identical fold split. (The
held-out `test.csv` is never touched — every cell is scored on it.)

## 5. Submit the matrix

```bash
mkdir -p logs/slurm results/composition
sbatch slurm/run_composition_matrix.sh
squeue -u "$USER"
```

`--array=0-74%4` runs up to 4 cells at once; lower the `%4` if your GPU quota is
smaller, raise it if you have more H200s.

Optional single-cell dry run first (a few min on 1 GPU):

```bash
python scripts/run_composition_experiment.py --arch unetpp --composition dfu_only \
    --fold 0 --seed 42 --epochs 2 --device cuda --num-workers 8
ls results/composition/   # expect unetpp_dfu_only_seed42_fold0.json
```

## 6. Monitor

```bash
squeue -u "$USER"
ls results/composition/*.json | wc -l        # completed cells (target: 75)
tail -f logs/slurm/*_0_composition.out
```

## 7. Aggregate into the paper table (CPU — login node is fine)

```bash
python scripts/aggregate_composition_results.py \
    --results-dir results/composition \
    --output-json results/composition_comparison.json \
    --output-md   results/composition_comparison.md
cat results/composition_comparison.md
```

The `.md` is the fold-averaged results table (DFU Dice mean ± std per
composition × architecture); the `.json` also carries per-cell numbers and the
paired-bootstrap significance of each composition vs DFU-only.

## 8. Bring the results back

Result artifacts are tiny (JSON + markdown + a few mask PNGs). `results/` is
gitignored, so **force-add** — these are the paper's evidence:

```bash
git add -f results/composition results/composition_comparison.json \
           results/composition_comparison.md data/splits/cv/folds_manifest.json
git add data/metadata/leakage_report_composition.json   # data/metadata is tracked
git commit -m "results(composition): clean-split CV composition study (75 cells)"
git push origin main     # push as the repo-owner account
```

---

## Notes / gotchas

- **Pre-warm ImageNet/DINOv2 weights** on a login node (internet) if compute
  nodes are offline:
  `python -c "import segmentation_models_pytorch as smp; smp.Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet'); smp.Segformer(encoder_name='mit_b0', encoder_weights='imagenet')"`
  and `python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')"`.
- **Wall-clock:** 75 cells at `%4` is ~1–1.5 days; DINOv2 (frozen backbone) cells
  are fast, U-Net++/SegFormer 50-epoch cells are the long pole. Raise `%` if quota allows.
- **Patient grouping caveat (state in the paper):** the public sources lack true
  patient IDs, so folds are grouped by the strongest available image-provenance id.
  Because every fold is scored on the fixed clean test set, the reported metric is
  unaffected by any residual intra-pool fold leakage.
- **Reproducibility:** each cell's JSON carries the test-split SHA, checkpoint SHA,
  seed, fold, git commit, input size, and per-class train counts.
