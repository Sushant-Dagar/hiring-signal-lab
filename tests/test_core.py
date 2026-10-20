import numpy as np
import pandas as pd
import pytest

from hsl import causal
from hsl.data import fingerprint, mask_title_words, normalise
from hsl.experiments import mcnemar_exact, mcnemar_sample_size, paired_bootstrap_ci
from hsl.metrics import average_precision, ndcg_at_k, precision_at_k, reciprocal_rank


# ------------------------------------------------------------------ data
def test_normalise_and_fingerprint_ignore_case_and_punctuation():
    assert normalise("  Java-Developer!! ﻿C++ ") == "java developer c++"
    assert fingerprint("Python, SQL") == fingerprint("python   sql")


def test_title_masking_removes_every_role_word():
    texts = pd.Series(["senior java developer with sql skills", "accountant handling ledgers"])
    out = mask_title_words(texts, ["Java Developer", "SQL Developer", "Accountant"])
    assert out.tolist() == ["senior with skills", "handling ledgers"]


# ------------------------------------------------------------------ metrics
def test_ranking_metrics_on_known_case():
    rel = np.array([0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.1, 0.7])  # ranking: 0,1,3,2 -> rel 0,1,1,0
    assert reciprocal_rank(rel, scores) == pytest.approx(0.5)
    assert precision_at_k(rel, scores, 2) == pytest.approx(0.5)
    assert average_precision(rel, scores) == pytest.approx((1 / 2 + 2 / 3) / 2)
    ideal = 1 + 1 / np.log2(3)
    assert ndcg_at_k(rel, scores, 4) == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / ideal)


def test_perfect_ranking_scores_one():
    rel = np.array([1, 1, 0, 0])
    s = np.array([4.0, 3.0, 2.0, 1.0])
    assert ndcg_at_k(rel, s, 10) == pytest.approx(1.0)
    assert average_precision(rel, s) == pytest.approx(1.0)


# ------------------------------------------------------------------ experiments
def test_mcnemar_counts_only_discordant_pairs():
    a = np.array([True, True, True, False, False])
    b = np.array([True, False, False, False, True])
    r = mcnemar_exact(a, b)
    assert (r["a_right_b_wrong"], r["b_right_a_wrong"]) == (2, 1)


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.8, 4000).astype(bool)
    b = rng.binomial(1, 0.7, 4000).astype(bool)
    m, lo, hi = paired_bootstrap_ci(a, b, n_boot=2000)
    assert lo < 0.10 < hi and lo > 0


def test_sample_size_shrinks_for_bigger_effects():
    assert mcnemar_sample_size(0.2, 0.02) < mcnemar_sample_size(0.2, 0.01)


# ------------------------------------------------------------------ causal
@pytest.fixture(scope="module")
def covariates():
    rng = np.random.default_rng(0)
    return rng.beta(5, 2, 3000), rng.lognormal(6.3, 0.6, 3000)


def test_naive_estimate_is_biased_upward_by_confounding(covariates):
    f = causal.simulate_funnel(*covariates, np.random.default_rng(3))
    assert causal.naive(f) > f.true_ate + 0.1


def test_adjusted_estimators_recover_planted_effect(covariates):
    errs = {k: [] for k in ("g", "ipw", "aipw")}
    rng = np.random.default_rng(5)
    for _ in range(40):
        f = causal.simulate_funnel(*covariates, rng)
        errs["g"].append(causal.g_computation(f) - f.true_ate)
        errs["ipw"].append(causal.ipw(f) - f.true_ate)
        errs["aipw"].append(causal.aipw(f)[0] - f.true_ate)
    for k, v in errs.items():
        assert abs(np.mean(v)) < 0.02, k


def test_ipw_weights_balance_covariates(covariates):
    f = causal.simulate_funnel(*covariates, np.random.default_rng(9))
    e = causal.propensity(f)
    w = np.where(f.t == 1, 1 / e, 1 / (1 - e))
    assert np.abs(causal.smd(f.X, f.t)).max() > 0.5
    assert np.abs(causal.smd(f.X, f.t, w)).max() < 0.1
