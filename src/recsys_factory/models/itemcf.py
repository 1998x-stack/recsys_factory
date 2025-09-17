from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, diags
from .base import BaseModel
from ..factory import register
from ..features.tfidf import item_tfidf
from ..log import logger

@register("model", "itemcf")
class ItemCF(BaseModel):
    """Cosine Item-based CF with optional TF-IDF and shrinkage."""
    def __init__(self, k: int = 200, shrinkage: float = 50.0, use_tfidf: bool = True):
        self.k = k; self.shrink = shrinkage; self.use_tfidf = use_tfidf
        self.S = None  # item-item similarity

    def fit(self, R: csr_matrix):
        logger.info(f"[model:itemcf] fitting k={self.k} shrink={self.shrink} tfidf={self.use_tfidf}")
        X = item_tfidf(R) if self.use_tfidf else R.tocsr().astype(float)
        norms = np.sqrt(X.multiply(X).sum(axis=0)).A.ravel() + 1e-8
        Xn = X @ diags(1.0 / norms)
        S = (Xn.T @ Xn).tocsr()
        if self.shrink > 0:
            counts = (R.T @ R).tocsr()
            S = S.multiply(counts / (counts + self.shrink))
        self.S = S
        logger.info("[model:itemcf] similarity computed")
        return self

    def score_user(self, u: int, R: csr_matrix, exclude_seen: bool = True) -> np.ndarray:
        assert self.S is not None
        s, e = R.indptr[u], R.indptr[u+1]
        items, vals = R.indices[s:e], R.data[s:e]
        scores = np.zeros(R.shape[1], dtype=float)
        for it, v in zip(items, vals):
            row = self.S.getrow(it)
            if self.k is not None and row.nnz > self.k:
                row = _topk_row(row, self.k)
            scores[row.indices] += row.data * v
        if exclude_seen: scores[items] = -np.inf
        return scores

def _topk_row(row: csr_matrix, k: int) -> csr_matrix:
    row = row.tocsr()
    if row.nnz <= k: return row
    data, idx = row.data, row.indices
    top = np.argpartition(-data, k)[:k]
    return csr_matrix((data[top], (np.zeros_like(top), idx[top])), shape=row.shape)
