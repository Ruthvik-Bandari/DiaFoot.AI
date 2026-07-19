# Composition Study — Explorer Runbook

Run the training-data-composition experiment on Northeastern Explorer, where the
images and GPUs live. Produces the paper's composition table: for each
(architecture × training composition × seed) cell, DFU-Dice with 95% CIs, the
false-positive rate on empty ground truth, and paired significance vs DFU-only —
all on the **fixed, clean test set**.

Every step is gated by the one before it. **Do not `sbatch` until the clean-split
gate (step 3) passes** — training on leaky splits reproduces the exact bug this
study exists to avoid.

---

## What runs (14 cells, `slurm/run_composition_matrix.sh`)

| Cells | Architecture | Composition | Seeds |
|---|---|---|---|
| 9 | U-Net++ (EffB4, scSE) | dfu_only · dfu+nonDFU · all | 41, 42, 43 |
| 2 | U-Net++ | negatives 25% · 50% of pool | 42 |
| 3 | DINOv2 ViT-B/14 (frozen + UPerNet) | dfu_only · dfu+nonDFU · all | 42 |

Each cell is independent, writes its filtered splits + checkpoint to its own dir,
and emits `results/composition/<run_tag>.json`.

---

## 0. Prereqs

- The composition-experiment code is on the branch Explorer will pull (the new
  `scripts/run_composition_experiment.py`, `scripts/aggregate_composition_results.py`,
  `src/data/composition.py`, `src/evaluation/composition_report.py`,
  `slurm/run_composition_matrix.sh`). **This requires the commit to be pushed to
  the GitHub repo first** (you curate `main` — see below).
- Clean, leak-free splits + images already on Explorer from the honest re-run:
  `data/splits/{train,val,test}.csv` and `data/processed/{dfu,healthy,non_dfu}/{images,masks}/`.
- The DINOv2 backbone is in the torch hub cache (`~/.cache/torch/hub/…`) from the
  prior DINOv2 training — the compute nodes may not have internet.

Set your repo location once (adjust to wherever the repo lives on Explorer):

```bash
export DIAFOOT=$HOME/DiaFoot.AI-v2      # or /scratch/$USER/DiaFoot.AI
cd "$DIAFOOT"
```

## 1. Get the code onto Explorer

```bash
git fetch origin
git checkout main
git pull --ff-only
git log --oneline -3      # expect the composition-experiment commit at/near the tip
ls scripts/run_composition_experiment.py slurm/run_composition_matrix.sh   # sanity
```

## 2. Load the environment (same modules/venv as the honest re-run)

```bash
source /etc/profile
module purge && module load cuda/12.8.0 python/3.13.5
source .venv/bin/activate
export PYTHONPATH="$DIAFOOT"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 3. GATE — prove the splits are the clean set (do NOT skip)

```bash
# 3a. Leakage audit must report has_any_leakage: false
python scripts/run_leakage_audit.py --splits-dir data/splits \
    --output data/metadata/leakage_report_composition.json --verbose
python - <<'PY'
import json
r = json.load(open("data/metadata/leakage_report_composition.json"))
flag = r.get("has_any_leakage", r.get("leakage", {}).get("has_any_leakage"))
assert flag is False, f"LEAKAGE PRESENT ({flag}) — STOP, rebuild clean splits first"
print("clean-split gate PASSED: has_any_leakage =", flag)
PY

# 3b. Record the exact test split this study is bound to (goes in the paper)
python - <<'PY'
import csv, hashlib, collections
rows = list(csv.DictReader(open("data/splits/test.csv")))
by = collections.Counter(r["class"] for r in rows)
sha = hashlib.sha256(open("data/splits/test.csv","rb").read()).hexdigest()
print(f"TEST n={len(rows)} by class={dict(by)}")
print(f"test.csv sha256={sha}")
PY
```

If 3a fails, stop and rebuild clean splits (see `docs/HPC_HONEST_RERUN_RUNBOOK.md`
step 2) before continuing.

## 4. Submit the matrix

```bash
mkdir -p logs/slurm results/composition
sbatch slurm/run_composition_matrix.sh
squeue -u "$USER"
```

`--array=0-13%4` runs up to 4 cells at once; lower the `%4` if your GPU quota is
smaller, raise it if you have more H200s.

Optional single-cell dry run before the full array (a few min on 1 GPU, 2 epochs):

```bash
python scripts/run_composition_experiment.py --arch unetpp --composition dfu_only \
    --seed 42 --epochs 2 --device cuda --num-workers 8
ls results/composition/   # expect unetpp_dfu_only_seed42.json
```

## 5. Monitor

```bash
squeue -u "$USER"
tail -f logs/slurm/*_0_composition.out      # a running cell
ls results/composition/*.json | wc -l       # completed cells (target: 14)
```

## 6. Aggregate into the paper table (CPU — login node is fine)

```bash
python scripts/aggregate_composition_results.py \
    --results-dir results/composition \
    --output-json results/composition_comparison.json \
    --output-md   results/composition_comparison.md
cat results/composition_comparison.md
```

## 7. Bring the results back

The result artifacts are tiny (JSON + markdown). Commit them from Explorer and
pull on the Mac (works even when interactive SSH/rsync is flaky). NOTE: `results/`
is gitignored, so the composition outputs must be **force-added** (`-f`) — they are
the paper's evidence and belong under version control:

```bash
git add -f results/composition results/composition_comparison.json \
           results/composition_comparison.md
git add data/metadata/leakage_report_composition.json   # data/metadata is tracked
git commit -m "results(composition): clean-split composition study (14 cells)"
git push origin main     # push as the repo-owner account
```

---

## Notes / gotchas

- **Push mechanics:** the repo owner is the personal GitHub account; push from the
  account that owns `Ruthvik-Bandari/DiaFoot.AI`.
- **DINOv2 offline:** if a DINOv2 cell fails to load the backbone, pre-warm the hub
  cache on a login node (which has internet):
  `python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')"`.
- **Reproducibility:** each cell's JSON carries the test-split SHA, checkpoint SHA,
  seed, git commit, and per-class train counts — cite these in the methods section.
- **Fair comparison:** every cell is evaluated on the *same* full clean test set;
  the DFU-only numbers are the DFU slice of that identical set.
