"""DiaFoot.AI v2 — External validation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap CI for the sample mean."""
    if len(values) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    n = len(values)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots.append(float(values[idx].mean()))

    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1.0 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
    }


def compute_drop_report(
    internal: dict[str, float],
    external: dict[str, float],
    keys: list[str],
) -> dict[str, Any]:
    """Compute absolute/relative drop for selected metrics."""
    out: dict[str, Any] = {}
    for key in keys:
        i = float(internal.get(key, 0.0))
        e = float(external.get(key, 0.0))
        abs_drop = i - e
        rel_drop = abs_drop / i if i != 0 else 0.0
        out[key] = {
            "internal": i,
            "external": e,
            "absolute_drop": abs_drop,
            "relative_drop": rel_drop,
        }
    return out
