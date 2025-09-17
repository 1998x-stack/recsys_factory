from __future__ import annotations
import numpy as np

def logloss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    # Mann-Whitney U statistic
    idx = np.argsort(scores)
    ranks = np.empty_like(idx, dtype=float)
    ranks[idx] = np.arange(1, len(scores) + 1)
    pos = (y_true == 1)
    n_pos = pos.sum(); n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.0
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return float(auc)
