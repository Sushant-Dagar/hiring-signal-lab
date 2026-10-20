"""Ranking and classification metrics.

The matching task is framed the way a recruiter sees it: for each job role
(the "query"), rank every held-out resume by how well it fits, and check
whether the true candidates for that role come out on top.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def dcg(rels: np.ndarray) -> float:
    rels = np.asarray(rels, dtype=float)
    return float(np.sum(rels / np.log2(np.arange(2, rels.size + 2))))


def ndcg_at_k(relevant: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    ideal = np.sort(relevant)[::-1][:k]
    best = dcg(ideal)
    return dcg(relevant[order]) / best if best > 0 else 0.0


def precision_at_k(relevant: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    return float(relevant[order].mean())


def reciprocal_rank(relevant: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = np.flatnonzero(relevant[order])
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


def average_precision(relevant: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    rel = relevant[order].astype(float)
    if rel.sum() == 0:
        return 0.0
    cum = np.cumsum(rel)
    return float(np.sum(rel * cum / np.arange(1, rel.size + 1)) / rel.sum())


def per_query_ranking(score_matrix: np.ndarray, y_true: np.ndarray, k: int = 10) -> dict[str, np.ndarray]:
    """score_matrix[i, c] = model's fit score of resume i for role c. Returns one value per role."""
    out = {f"ndcg@{k}": [], f"p@{k}": [], "mrr": [], "ap": []}
    for c in range(score_matrix.shape[1]):
        rel = (y_true == c).astype(int)
        s = score_matrix[:, c]
        out[f"ndcg@{k}"].append(ndcg_at_k(rel, s, k))
        out[f"p@{k}"].append(precision_at_k(rel, s, k))
        out["mrr"].append(reciprocal_rank(rel, s))
        out["ap"].append(average_precision(rel, s))
    return {m: np.array(v) for m, v in out.items()}


def classification(score_matrix: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    pred = score_matrix.argmax(1)
    top3 = np.argsort(-score_matrix, 1)[:, :3]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "top3_accuracy": float(np.mean([y in t for y, t in zip(y_true, top3)])),
    }


def summarise(score_matrix: np.ndarray, y_true: np.ndarray, k: int = 10) -> dict[str, float]:
    rank = per_query_ranking(score_matrix, y_true, k)
    res = {m: float(v.mean()) for m, v in rank.items()}
    res["map"] = res.pop("ap")
    res.update(classification(score_matrix, y_true))
    return res
