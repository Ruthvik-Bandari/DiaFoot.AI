# Honest Re-run on Northeastern Explorer — Runbook

Goal: produce **trustworthy** metrics after the leakage + correctness fixes, run
where the images and GPUs live. Every step is gated by the previous one — do not
skip ahead.

**Status of the code fixes (all already on `origin/main`):**
- Split-grouping leakage fix (`rebuild_splits_strict.py`: group by source-image id,
  then full-pairwise dHash near-dup merge).
- E1 metrics, D1 ITA `atan`, I1/I2 inference, D2/D3 leakage-audit, EMA buffer sync,
  TTA D4, scheduler warmup, and the cosine-progress clamp (T3).
- Interim source-image-grouped splits committed at `data/splits_grouped/` (near-dup
  merge still pending — it needs the images, which is step B below).

So Explorer only needs to **pull `main`** and run B–G. The push is done; nothing is
stuck on the Mac.

---

## 0. Prereqs on Explorer
- Repo cloned; image data present:
  - `data/processed/{dfu,healthy,non_dfu}/{images,masks}/*.png`  (split / train / eval)
  - `data/raw/…`  (ITA regeneration)
- `.venv` built and consistent with the module you load:
  ```bash
  module load cuda/12.8.0 python/3.12   # match train script's module if it differs
  python -m venv .venv && .venv/bin/pip install -r requirements.txt
  ```

## 1. Get the fixed code onto Explorer
```bash
git fetch origin
git checkout main
git pull --ff-only        # brings all honesty fixes + split-grouping fix + T3 + grouped splits
git log --oneline -3      # expect 7ce4aa7 (near-dup guard) at/near the tip
```

## 2. (B) Regenerate CLEAN splits WITH images — the dHash near-dup step now fires
```bash
cp -r data/splits data/splits_backup_$(date +%F)   # preserve the old leaky split as evidence
.venv/bin/python scripts/rebuild_splits_strict.py --split-dir data/splits --seed 42 --near-threshold 6
```
On the Mac (no images) this fixed the 60 source-image straddles → 0. **With images it
also merges cross-image perceptual near-duplicates** — the half that could not run on
the Mac. This overwrites `data/splits/{train,val,test}.csv` in place (the backup above
is your rollback). `data/splits_grouped/` was the interim source-grouping-only artifact;
after this step, `data/splits/` is the authoritative clean split.

## 3. (F) Prove the split is clean — content + near-dup checks actually execute now
```bash
.venv/bin/python scripts/run_leakage_audit.py --splits-dir data/splits \
    --output data/metadata/leakage_report.json --verbose
```
Expect path / canonical / content / near-duplicate overlaps **ALL = 0**. (On the Mac the
content + near-dup checks silently reported 0 because the image files were absent; here
they truly execute.)

## 4. (C) Retrain on the clean splits (GPU) — old checkpoints were trained on leaked data
```bash
sbatch slurm/train_dinov2.sh            # full pipeline (classifier + segmenter)
# or: sbatch slurm/train_dinov2.sh classify   /   segment
# → checkpoints/dinov2_classifier/best.pt , checkpoints/dinov2_segmenter/best.pt
```

## 5. (D) Re-evaluate (E1 fix applies) + external + repro bundle
`slurm/rerun_metrics_h200.sh` already points `CLASSIFIER_CKPT` / `SEGMENTER_CKPT` at
`checkpoints/dinov2_{classifier,segmenter}/best.pt`. Confirm the new run wrote `best.pt`
(else edit the two paths at the top), then:
```bash
sbatch slurm/rerun_metrics_h200.sh
# regenerates results/{classification_metrics,segmentation_metrics,external_validation,...}.json
```

## 6. (E) Regenerate ITA (D1 — atan fix; expect ~0 category changes for real skin)
```bash
.venv/bin/python scripts/run_ita_analysis.py --data-root data/raw --output-dir data/metadata
```

## 7. (G) Rewrite the honest numbers
Update `Midterm_Report_DiaFootAI_v2.md` from the regenerated `results/*.json`. **Expect
internal metrics to DROP** from the near-perfect 0.98–1.0 (they were inflated by leakage)
toward the external-validation reality.

## 8. Commit the honest artifacts back (as the personal account)
```bash
git add data/splits data/metadata results Midterm_Report_DiaFootAI_v2.md
git commit -m "data(honest-rerun): clean splits + retrained metrics + ITA + report"
git push origin main    # ensure you push as Ruthvik-Bandari (repo owner)
```

---

## Post-run checks / open items
- **External-validation collapse:** internal was 0.98–1.0 vs external 0.33. After the
  leakage fix + retrain, confirm internal **drops toward** external. If a large gap
  remains, audit the external split itself for cross-dataset leakage.
- **Patient-level leakage (out of scope, unclosed):** the fix closes *image-level*
  leakage. If one real patient has multiple photos under different ids, they can still
  straddle splits. Needs patient metadata we don't have.
- `external.csv` / `external_segmentation.csv` are a separate held-out dataset;
  `rebuild_splits_strict.py` does not touch them.
- If Explorer's `python` module differs (train uses 3.13.5, rerun uses 3.12), keep one
  venv consistent with the module you load.
