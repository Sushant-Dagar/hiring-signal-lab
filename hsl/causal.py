"""Causal effect estimation on a semi-synthetic hiring funnel.

Question: does a recruiter "fast-track" flag *cause* more candidates to be
shortlisted, or do fast-tracked candidates just happen to be stronger?

Real recruiting logs never reveal the true answer, so estimators cannot be
checked on them. Instead we build a funnel on top of the real held-out
resumes with a KNOWN effect planted in it, then see which estimators recover
it:

* covariates come from the real data: the matching model's out-of-sample fit
  score for the candidate's role, and resume length (an experience proxy);
* stronger candidates are more likely to be fast-tracked (confounding);
* stronger candidates are also more likely to be shortlisted regardless;
* fast-tracking adds a fixed boost on the log-odds scale.

Because both potential outcomes are simulated, the true average treatment
effect (ATE) is known exactly. The naive comparison is biased by the
confounding; outcome regression, inverse-propensity weighting and the doubly
robust AIPW estimator should recover the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Funnel:
    X: np.ndarray          # covariates (n, k), standardised
    t: np.ndarray          # treatment: fast-tracked (0/1)
    y: np.ndarray          # outcome: shortlisted (0/1)
    true_ate: float        # exact ATE on this sample (mean of p1 - p0)


def simulate_funnel(fit: np.ndarray, n_words: np.ndarray, rng: np.random.Generator,
                    tau_logit: float = 0.6, confounding: float = 1.6, nonlinear: float = 0.0) -> Funnel:
    z_fit = (fit - fit.mean()) / fit.std()
    lw = np.log(n_words)
    z_len = (lw - lw.mean()) / lw.std()
    X = np.column_stack([z_fit, z_len])

    p_treat = _sigmoid(-0.8 + confounding * z_fit + 0.4 * z_len)
    t = rng.binomial(1, p_treat)

    # `nonlinear` adds a threshold-like curve the linear outcome model cannot capture
    base = -0.6 + 1.4 * z_fit + 0.5 * z_len + nonlinear * (np.maximum(z_fit, 0) ** 2 - 0.5)
    p0, p1 = _sigmoid(base), _sigmoid(base + tau_logit)
    y = rng.binomial(1, np.where(t == 1, p1, p0))
    return Funnel(X=X, t=t, y=y, true_ate=float(np.mean(p1 - p0)))


# --------------------------------------------------------------- estimators
def naive(f: Funnel) -> float:
    return float(f.y[f.t == 1].mean() - f.y[f.t == 0].mean())


def _outcome_model(f: Funnel):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack([f.X, f.t]), f.y)
    ones, zeros = np.ones_like(f.t), np.zeros_like(f.t)
    mu1 = m.predict_proba(np.column_stack([f.X, ones]))[:, 1]
    mu0 = m.predict_proba(np.column_stack([f.X, zeros]))[:, 1]
    return mu0, mu1


def propensity(f: Funnel, clip: float = 0.01) -> np.ndarray:
    e = LogisticRegression(C=1e6, max_iter=1000).fit(f.X, f.t).predict_proba(f.X)[:, 1]
    return np.clip(e, clip, 1 - clip)


def g_computation(f: Funnel) -> float:
    mu0, mu1 = _outcome_model(f)
    return float(np.mean(mu1 - mu0))


def ipw(f: Funnel) -> float:
    """Stabilised (Hajek) inverse-propensity weighting."""
    e = propensity(f)
    w1, w0 = f.t / e, (1 - f.t) / (1 - e)
    return float(np.sum(w1 * f.y) / np.sum(w1) - np.sum(w0 * f.y) / np.sum(w0))


def aipw(f: Funnel) -> tuple[float, float]:
    """Doubly robust estimate and its influence-function standard error."""
    e = propensity(f)
    mu0, mu1 = _outcome_model(f)
    psi = (mu1 - mu0) + f.t * (f.y - mu1) / e - (1 - f.t) * (f.y - mu0) / (1 - e)
    return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(psi.size))


def smd(X: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Standardised mean difference per covariate (|SMD| < 0.1 is the usual balance bar)."""
    w = np.ones_like(t, dtype=float) if w is None else w
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
        v1 = np.average((x[t == 1] - m1) ** 2, weights=w[t == 1]); v0 = np.average((x[t == 0] - m0) ** 2, weights=w[t == 0])
        out.append((m1 - m0) / np.sqrt((v1 + v0) / 2))
    return np.array(out)


def bootstrap(f: Funnel, est, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = f.t.size
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(est(Funnel(f.X[i], f.t[i], f.y[i], f.true_ate)))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


ESTIMATORS = {
    "Naive difference": naive,
    "Outcome regression (g-computation)": g_computation,
    "Inverse propensity weighting": ipw,
    "Doubly robust (AIPW)": lambda f: aipw(f)[0],
}


def single_study(fit, n_words, seed: int = 7, n_boot: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    f = simulate_funnel(fit, n_words, rng)
    e = propensity(f)
    w = np.where(f.t == 1, 1 / e, 1 / (1 - e))
    res = {"n": int(f.t.size), "treated_share": float(f.t.mean()), "true_ate": f.true_ate,
           "smd_before": smd(f.X, f.t).tolist(), "smd_after_ipw": smd(f.X, f.t, w).tolist(), "estimates": {}}
    for name, est in ESTIMATORS.items():
        lo, hi = bootstrap(f, est, n_boot, rng)
        res["estimates"][name] = {"estimate": est(f), "ci95": [lo, hi]}
    return res


def repeated_studies(fit, n_words, n_sims: int = 300, seed: int = 11, nonlinear: float = 0.0) -> dict:
    """Bias, RMSE and (for AIPW) 95% CI coverage across many simulated funnels."""
    rng = np.random.default_rng(seed)
    errs = {k: [] for k in ESTIMATORS}
    covered = []
    for _ in range(n_sims):
        f = simulate_funnel(fit, n_words, rng, nonlinear=nonlinear)
        for name, est in ESTIMATORS.items():
            errs[name].append(est(f) - f.true_ate)
        a, se = aipw(f)
        covered.append(abs(a - f.true_ate) <= 1.96 * se)
    return {
        "n_sims": n_sims,
        "nonlinear": nonlinear,
        "bias": {k: float(np.mean(v)) for k, v in errs.items()},
        "rmse": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()},
        "aipw_ci_coverage": float(np.mean(covered)),
    }
