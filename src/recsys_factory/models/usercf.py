from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .base import BaseModel
from ..factory import register
from ..log import logger

@register("model", "usercf")
class UserCF(BaseModel):
    """Cosine User-based CF with shrinkage and top-k neighbors."""
    def __init__(self, k: int = 100, shrinkage: float = 10.0):
        self.k = k
        self.shrink = shrinkage
        self.S = None  # user-user similarity

    def fit(self, R: csr_matrix):
        logger.info(f"[model:usercf] fitting k={self.k} shrink={self.shrink}")
        R = R.tocsr().astype(float)
        norms = np.sqrt(R.multiply(R).sum(axis=1)).A.ravel() + 1e-8
        Rn = R.multiply(1.0 / norms[:, None])
        S = (Rn @ Rn.T).tocsr()
        if self.shrink > 0:
            counts = (R @ R.T).tocsr()
            S = S.multiply(counts / (counts + self.shrink))
        self.S = S
        logger.info("[model:usercf] similarity computed")
        return self

    def score_user(self, u: int, R: csr_matrix, exclude_seen: bool = True) -> np.ndarray:
        assert self.S is not None
        Su = self.S.getrow(u)
        if self.k is not None and self.k < self.S.shape[0]:
            Su = _topk_row(Su, self.k)
        scores = Su @ R
        scores = np.asarray(scores.todense()).ravel()
        if exclude_seen:
            s, e = R.indptr[u], R.indptr[u+1]
            scores[R.indices[s:e]] = -np.inf
        return scores

def _topk_row(row: csr_matrix, k: int) -> csr_matrix:
    row = row.tocsr()
    if row.nnz <= k: return row
    data, idx = row.data, row.indices
    top = np.argpartition(-data, k)[:k]
    return csr_matrix((data[top], (np.zeros_like(top), idx[top])), shape=row.shape)
