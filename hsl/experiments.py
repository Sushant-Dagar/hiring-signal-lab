"""Head-to-head (A/B) comparison of two scorers evaluated on the same candidates.

Because both models score exactly the same held-out resumes, the comparison is
*paired*: we test the per-item difference, not two independent averages. That
removes the between-resume noise and gives a much sharper test.

* Top-1 correctness  -> exact McNemar test on the discordant pairs, plus a
  paired bootstrap 95% CI for the accuracy difference.
* Per-role ranking quality (AP, NDCG@10) -> Wilcoxon signed-rank across the
  43 roles, plus a paired bootstrap CI over roles.
* Power: how many evaluation resumes are needed to detect a given accuracy gap,
  using the observed discordance rate (McNemar sample-size formula).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """CI for mean(a - b) resampling the paired items."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    p = float(stats.binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"a_right_b_wrong": a_only, "b_right_a_wrong": b_only, "p_value": p}


def mcnemar_sample_size(p_discordant: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items needed to detect an accuracy difference `delta` given the discordance rate (Connor 1987)."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    n = (za * np.sqrt(p_discordant) + zb * np.sqrt(p_discordant - delta**2)) ** 2 / delta**2
    return int(np.ceil(n))


def compare(scores_a: np.ndarray, scores_b: np.ndarray, y: np.ndarray,
            per_query_a: dict, per_query_b: dict, seed: int = 0) -> dict:
    ca = scores_a.argmax(1) == y
    cb = scores_b.argmax(1) == y
    acc = paired_bootstrap_ci(ca, cb, seed=seed)
    out = {
        "accuracy_a": float(ca.mean()),
        "accuracy_b": float(cb.mean()),
        "accuracy_diff": {"mean": acc[0], "ci95": [acc[1], acc[2]]},
        "mcnemar": mcnemar_exact(ca, cb),
    }
    for metric in ("ap", "ndcg@10"):
        qa, qb = per_query_a[metric], per_query_b[metric]
        m, lo, hi = paired_bootstrap_ci(qa, qb, seed=seed)
        nonzero = np.abs(qa - qb) > 1e-12
        w = stats.wilcoxon(qa[nonzero], qb[nonzero]) if nonzero.sum() >= 6 else None
        out[f"{metric}_per_role"] = {
            "mean_diff": m, "ci95": [lo, hi],
            "a_better_roles": int(np.sum(qa > qb + 1e-12)), "b_better_roles": int(np.sum(qb > qa + 1e-12)),
            "wilcoxon_p": float(w.pvalue) if w is not None else None,
        }
    disc = float(np.mean(ca != cb))
    out["power"] = {
        "discordance_rate": disc,
        "n_needed_for_1pp": mcnemar_sample_size(disc, 0.01),
        "n_needed_for_2pp": mcnemar_sample_size(disc, 0.02),
        "n_test": int(y.size),
    }
    return out
