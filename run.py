"""End-to-end pipeline: clean data, train three matchers, evaluate, A/B test, causal study, charts.

    python run.py            # full run (about 8 minutes on a laptop CPU)
    python run.py --quick    # 2-epoch smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsl import causal
from hsl.data import class_weights, load_clean, mask_title_words, split
from hsl.experiments import compare
from hsl.metrics import per_query_ranking, summarise
from hsl.models import TfidfLogReg, TorchBagMatcher, TorchTfidfMLP

ROOT = Path(__file__).parent
FIG = ROOT / "reports" / "figures"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
COLORS = ["#2563eb", "#f59e0b", "#10b981", "#ef4444"]

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main(quick: bool = False):
    t_start = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # ---------------------------------------------------------------- data
    df, rep = load_clean(ROOT / "data" / "raw" / "resumes.csv")
    train, val, test = split(df)
    roles = sorted(df["role"].unique())
    idx = {r: i for i, r in enumerate(roles)}
    ytr, yva, yte = (d["role"].map(idx).to_numpy() for d in (train, val, test))
    cw = class_weights(ytr, len(roles))
    print(f"Data: {rep.as_dict()}  |  train {len(train)}  val {len(val)}  test {len(test)}  roles {len(roles)}")

    epochs = 2 if quick else 30

    # ---------------------------------------------------------------- models
    models = {}
    print("Training TF-IDF + Logistic Regression ...")
    models["lr"] = TfidfLogReg().fit(train["text"], ytr)
    print("Training PyTorch MLP on TF-IDF ...")
    models["mlp"] = TorchTfidfMLP(len(roles), epochs=epochs).fit(train["text"], ytr, val["text"], yva, cw)
    print("Training PyTorch EmbeddingBag network ...")
    models["bag"] = TorchBagMatcher(len(roles), epochs=epochs).fit(train["text"], ytr, val["text"], yva, cw)

    scores = {k: m.scores(test["text"]) for k, m in models.items()}
    results = {models[k].name: summarise(s, yte) for k, s in scores.items()}
    per_query = {k: per_query_ranking(s, yte) for k, s in scores.items()}
    for name, r in results.items():
        print(f"  {name:32s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  MAP {r['map']:.3f}  NDCG@10 {r['ndcg@10']:.3f}")

    # ---------------------------------------------------------------- title-leak ablation
    print("Title-word ablation: retraining the baseline with every role-name word removed ...")
    masked_train = mask_title_words(train["text"], roles)
    masked_test = mask_title_words(test["text"], roles)
    lr_masked = TfidfLogReg().fit(masked_train, ytr)
    masked_scores = lr_masked.scores(masked_test)
    ablation = {"full_text": results[models["lr"].name], "title_words_removed": summarise(masked_scores, yte)}
    print(f"  accuracy {ablation['full_text']['accuracy']:.3f} -> {ablation['title_words_removed']['accuracy']:.3f}")

    # ---------------------------------------------------------------- error analysis
    pred = scores["lr"].argmax(1)
    conf = pd.crosstab(pd.Series(yte, name="true"), pd.Series(pred, name="pred"))
    pairs = []
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            n = conf.get(j, {}).get(i, 0) + conf.get(i, {}).get(j, 0)
            if n:
                pairs.append((roles[i], roles[j], int(n)))
    pairs.sort(key=lambda p: -p[2])
    per_role = pd.DataFrame({"role": roles, "ap": per_query["lr"]["ap"], "ndcg@10": per_query["lr"]["ndcg@10"],
                             "n_test": np.bincount(yte, minlength=len(roles))}).sort_values("ap")
    per_role.to_csv(ROOT / "reports" / "per_role_baseline.csv", index=False)

    # ---------------------------------------------------------------- A/B test
    print("A/B test: Logistic Regression (A) vs PyTorch MLP (B) on the same test resumes ...")
    ab = compare(scores["lr"], scores["mlp"], yte, per_query["lr"], per_query["mlp"])
    print(f"  accuracy diff {ab['accuracy_diff']['mean']:+.4f}  CI {ab['accuracy_diff']['ci95']}  McNemar p={ab['mcnemar']['p_value']:.2g}")

    # ---------------------------------------------------------------- causal study
    print("Causal study on the semi-synthetic funnel ...")
    fit = scores["lr"][np.arange(len(yte)), yte]
    nw = test["n_words"].to_numpy()
    study = causal.single_study(fit, nw, n_boot=100 if quick else 500)
    sims = causal.repeated_studies(fit, nw, n_sims=30 if quick else 300)
    sims_mis = causal.repeated_studies(fit, nw, n_sims=30 if quick else 300, nonlinear=1.5)
    print(f"  true ATE {study['true_ate']:.3f}  naive {study['estimates']['Naive difference']['estimate']:.3f}  "
          f"AIPW {study['estimates']['Doubly robust (AIPW)']['estimate']:.3f}  coverage {sims['aipw_ci_coverage']:.2f}")

    # ---------------------------------------------------------------- figures
    names = [models[k].name for k in ("lr", "mlp", "bag")]
    mets = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("map", "MAP"), ("ndcg@10", "NDCG@10")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(mets)); w = 0.26
    for i, n in enumerate(names):
        vals = [results[n][m] for m, _ in mets]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=n, color=COLORS[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [l for _, l in mets]); ax.set_ylim(0.5, 1.05); ax.set_title("Resume-to-role matching on held-out resumes")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3); ax.grid(axis="x", visible=False)
    save(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    pr = per_role.sort_values("ap")
    col = [COLORS[3] if v < 0.75 else COLORS[0] for v in pr["ap"]]
    ax.barh(pr["role"], pr["ap"], color=col)
    ax.axvline(pr["ap"].mean(), color=INK, lw=1, ls="--"); ax.text(pr["ap"].mean() + 0.005, -0.9, f"mean {pr['ap'].mean():.2f}", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Average precision when ranking all test resumes for the role")
    ax.set_title("Where the baseline matcher struggles (red: AP < 0.75)"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "per_role_ap.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    top = pairs[:10][::-1]
    ax.barh([f"{a}  <->  {b}" for a, b, _ in top], [n for *_, n in top], color=COLORS[1])
    ax.set_xlabel("Test resumes confused between the two roles (both directions)")
    ax.set_title("Most-confused role pairs"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8)
    save(fig, "confused_pairs.png")

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    rows = [("Accuracy (per resume)", ab["accuracy_diff"]["mean"], ab["accuracy_diff"]["ci95"]),
            ("Average precision (per role)", ab["ap_per_role"]["mean_diff"], ab["ap_per_role"]["ci95"]),
            ("NDCG@10 (per role)", ab["ndcg@10_per_role"]["mean_diff"], ab["ndcg@10_per_role"]["ci95"])]
    for i, (lbl, m, (lo, hi)) in enumerate(rows):
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=COLORS[0], capsize=4, ms=6)
        ax.text(hi + 0.002, i, f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=COLORS[3], lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Logistic Regression minus PyTorch MLP (paired bootstrap 95% CI)")
    ax.set_title("A/B test: does the neural network beat the baseline?")
    lo_all = min(r[2][0] for r in rows); hi_all = max(r[2][1] for r in rows)
    ax.set_xlim(min(lo_all, 0) - 0.01, hi_all + 0.06)
    save(fig, "ab_test.png")

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    est = study["estimates"]
    for i, (name, e) in enumerate(est.items()):
        lo, hi = e["ci95"]
        c = COLORS[3] if name.startswith("Naive") else COLORS[0]
        ax.errorbar(e["estimate"], i, xerr=[[e["estimate"] - lo], [hi - e["estimate"]]], fmt="o", color=c, capsize=4, ms=6)
    ax.axvline(study["true_ate"], color=COLORS[2], lw=2, label=f"true effect {study['true_ate']:.3f}")
    ax.set_yticks(range(len(est)), list(est)); ax.invert_yaxis()
    ax.set_xlabel("Estimated effect of fast-tracking on shortlist rate (95% bootstrap CI)")
    ax.set_title("Causal estimate vs the planted true effect"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "causal_estimates.png")

    h = pd.DataFrame(models["mlp"].history)
    fig, ax1 = plt.subplots(figsize=(6.5, 3.0))
    ax1.plot(h["epoch"], h["train_loss"], color=COLORS[0], marker="o", ms=3, label="train loss")
    ax2 = ax1.twinx(); ax2.plot(h["epoch"], h["val_macro_f1"], color=COLORS[1], marker="o", ms=3, label="val macro-F1")
    ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax1.axvline(models["mlp"].best_epoch, color=MUTED, ls="--", lw=1)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax2.set_ylabel("validation macro-F1")
    ax1.set_title("PyTorch MLP learning curve (dashed: early-stopping epoch)")
    save(fig, "mlp_learning_curve.png")

    # ---------------------------------------------------------------- write out
    out = {
        "data": {**rep.as_dict(), "train": len(train), "val": len(val), "test": len(test), "roles": len(roles)},
        "models": results,
        "mlp_best_epoch": models["mlp"].best_epoch,
        "bag_best_epoch": models["bag"].best_epoch,
        "title_word_ablation": ablation,
        "top_confused_pairs": [{"a": a, "b": b, "n": n} for a, b, n in pairs[:10]],
        "weakest_roles": per_role.head(6).to_dict("records"),
        "ab_test_lr_vs_mlp": ab,
        "causal": {"single_study": study, "repeated": sims, "repeated_misspecified_outcome": sims_mis},
        "runtime_s": round(time.time() - t_start, 1),
        "quick": quick,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"Done in {out['runtime_s']} s -> reports/metrics.json, reports/figures/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
