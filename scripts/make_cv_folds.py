"""DiaFoot.AI — generate shared cross-validation folds for the composition study.

Partitions the train+val pool (NOT the held-out test set) into grouped k-folds
ONCE, so every composition x architecture cell trains/validates on the identical
fold partition. Each fold is the full pool (all classes); the per-cell
composition filter is applied later by run_composition_experiment.py.

Folds are grouped by the strongest available provenance id (``patient_id``,
which in these public datasets is ~unique per image — see group_kfold's note).
The held-out test set is untouched: every fold is evaluated on it, so the
reported metric is comparable across all cells and immune to intra-pool fold
leakage.

Usage:
    python scripts/make_cv_folds.py --n-folds 5 --seed 42
    # -> data/splits/cv/fold{0..4}/{train,val}.csv + folds_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.composition import class_counts, group_kfold, read_split_csv, write_split_csv
from src.evaluation.composition_report import git_commit, sha256_file

logger = logging.getLogger("make_cv_folds")


def main() -> None:
    """Generate and persist the shared CV folds + a provenance manifest."""
    parser = argparse.ArgumentParser(description="Generate shared CV folds")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--cv-dir", default="data/splits/cv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-key", default="patient_id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    splits = Path(args.splits_dir)
    train_csv, val_csv = splits / "train.csv", splits / "val.csv"
    pool = read_split_csv(train_csv) + read_split_csv(val_csv)
    logger.info("pool (train+val): %d rows, classes=%s", len(pool), class_counts(pool))

    folds = group_kfold(pool, n_folds=args.n_folds, seed=args.seed, group_key=args.group_key)

    cv_dir = Path(args.cv_dir)
    fold_records = []
    for i, (train_rows, val_rows) in enumerate(folds):
        fold_dir = cv_dir / f"fold{i}"
        write_split_csv(train_rows, fold_dir / "train.csv")
        write_split_csv(val_rows, fold_dir / "val.csv")
        n_val_groups = len({r.get(args.group_key) or r["image"] for r in val_rows})
        rec = {
            "fold": i,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "train_by_class": class_counts(train_rows),
            "val_by_class": class_counts(val_rows),
            "n_val_groups": n_val_groups,
        }
        fold_records.append(rec)
        logger.info(
            "fold %d: train=%d val=%d (val groups=%d)",
            i,
            len(train_rows),
            len(val_rows),
            n_val_groups,
        )

    manifest = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "group_key": args.group_key,
        "pool_size": len(pool),
        "source_train_csv": str(train_csv),
        "source_val_csv": str(val_csv),
        "source_train_sha256": sha256_file(train_csv),
        "source_val_sha256": sha256_file(val_csv),
        "git_commit": git_commit(splits if splits.exists() else None),
        "folds": fold_records,
    }
    manifest_path = cv_dir / "folds_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %d folds -> %s and %s", args.n_folds, cv_dir, manifest_path)


if __name__ == "__main__":
    main()
