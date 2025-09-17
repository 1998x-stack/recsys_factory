from __future__ import annotations
import numpy as np

def _dcg(rels: np.ndarray) -> float:
    idx = np.arange(1, rels.size + 1)
    return float((rels / np.log2(idx + 1)).sum())

def precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    k = min(k, pred.size)
    hits = np.intersect1d(pred[:k], truth).size
    return hits / max(1, k)

def recall_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    if truth.size == 0: return 0.0
    k = min(k, pred.size)
    hits = np.intersect1d(pred[:k], truth).size
    return hits / truth.size

def ndcg_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    k = min(k, pred.size)
    rels = np.isin(pred[:k], truth).astype(float)
    dcg = _dcg(rels)
    idcg = _dcg(np.sort(rels)[::-1])
    return 0.0 if idcg == 0 else dcg / idcg

def map_at_k(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    k = min(k, pred.size)
    ap = 0.0; hits = 0
    for i, p in enumerate(pred[:k], 1):
        if p in truth:
            hits += 1
            ap += hits / i
    return 0.0 if hits == 0 else ap / min(k, truth.size)
