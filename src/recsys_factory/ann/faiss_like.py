from __future__ import annotations
import numpy as np

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size: return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]
