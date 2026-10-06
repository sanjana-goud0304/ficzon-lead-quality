"""Evaluation helpers: ranking-quality metrics and bootstrap intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _ranked_order(score: np.ndarray, seed: int) -> np.ndarray:
    """Indices sorted by descending score; exact ties are broken randomly (seeded), never by row order."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(score))
    return perm[np.argsort(-score[perm], kind="stable")]


def topk_flags(score: np.ndarray, frac: float = 0.2, seed: int = 42) -> np.ndarray:
    """0/1 vector flagging the top `frac` of leads by score (ties broken randomly, seeded)."""
    score = np.asarray(score, dtype=float)
    k = max(1, int(round(frac * len(score))))
    flags = np.zeros(len(score), dtype=int)
    flags[_ranked_order(score, seed)[:k]] = 1
    return flags


def topk_metrics(y: np.ndarray, score: np.ndarray, frac: float = 0.2, seed: int = 42) -> dict[str, float]:
    """Precision, recall and lift if reps work only the top `frac` of leads by score."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    k = max(1, int(round(frac * len(y))))
    top = _ranked_order(score, seed)[:k]
    precision = float(y[top].mean())
    recall = float(y[top].sum() / y.sum()) if y.sum() else float("nan")
    return {"precision_at_k": precision, "recall_at_k": recall, "lift_at_k": precision / float(y.mean())}


def bootstrap_ci(y: np.ndarray, score: np.ndarray, metric, n_boot: int = 300, seed: int = 42) -> tuple[float, float]:
    """95% percentile interval of metric(y, score) over resampled rows."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        if y[b].min() != y[b].max():
            vals.append(metric(y[b], score[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion matrix and precision/recall/F1 when leads with score >= threshold are flagged High."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "precision": precision, "recall": recall, "f1": f1,
            "flag_rate": float(pred.mean())}


def evaluate_scores(y: np.ndarray, score: np.ndarray, frac: float = 0.2, seed: int = 42) -> dict[str, float]:
    lo, hi = bootstrap_ci(y, score, roc_auc_score, seed=seed)
    plo, phi = bootstrap_ci(y, score, average_precision_score, seed=seed)
    out = {"roc_auc": roc_auc_score(y, score), "roc_lo": lo, "roc_hi": hi,
           "pr_auc": average_precision_score(y, score), "pr_lo": plo, "pr_hi": phi}
    out.update(topk_metrics(y, score, frac, seed))
    return out
