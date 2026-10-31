from __future__ import annotations

import numpy as np


def paired_bootstrap_difference(a, b, *, n_boot=1000, seed=42, statistic="mean"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have same shape, got {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("Cannot bootstrap empty arrays")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    n = a.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(b[idx]) - np.mean(a[idx])
    return {
        "observed_difference": float(np.mean(b) - np.mean(a)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "n_boot": int(n_boot),
    }
