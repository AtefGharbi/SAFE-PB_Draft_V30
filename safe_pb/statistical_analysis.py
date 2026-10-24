"""Statistical protocol (manuscript Sec. 6.4): paired tests, bootstrap CIs,
Holm-corrected multiple comparisons, and effect sizes. These operate on
per-seed arrays. The publication runner enforces at least 20 independent
training seeds unless its explicit smoke-test override is supplied."""
from __future__ import annotations
import numpy as np
from scipy import stats


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return {"mean": float(x.mean()) if len(x) else float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "note": "insufficient seeds for a bootstrap CI (need >= 2)"}
    # Vectorize the bootstrap resampling. The previous Python loop made final
    # report generation and resumed runs spend tens of seconds in statistics
    # even after every expensive experiment job had already completed.
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[indices].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def paired_test(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return {"test": "none", "p_value": float("nan"),
                "note": "need >=3 paired seeds for a nonparametric paired test"}
    differences = x - y
    if np.allclose(differences, 0.0):
        return {"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0,
                "note": "all paired differences are zero"}
    stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return {"test": "wilcoxon", "statistic": float(stat), "p_value": float(p)}


def holm_correction(p_values: list) -> list:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Independent-samples Cohen's d (pooled variance). Use this ONLY when
    x and y come from genuinely independent samples (e.g. different,
    unpaired seeds). For paired comparisons -- the manuscript's actual
    protocol, Sec. 6.4: "pair methods by identical test-day and
    uncertainty seeds" -- use cohens_dz instead. Using pooled-variance d on
    paired data was flagged as wrong by external review; both functions
    are now provided so the caller cannot accidentally reach for the wrong
    one without noticing this docstring."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pooled_std = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) /
                          max(len(x) + len(y) - 2, 1)) if len(x) > 1 and len(y) > 1 else np.nan
    if not pooled_std or np.isnan(pooled_std) or pooled_std == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled_std)


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d_z for PAIRED samples: mean of the per-pair differences
    divided by the standard deviation of those differences. This is the
    correct effect size for the manuscript's protocol (same seed/test-day
    pairs across methods), not the independent-samples cohens_d above."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    diff = x - y
    sd = diff.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(diff.mean() / sd)
