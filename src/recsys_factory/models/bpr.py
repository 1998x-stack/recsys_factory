from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .base import BaseModel
from ..factory import register
from ..log import logger

@register("model", "bpr")
class BPR(BaseModel):
    """Pairwise BPR with SGD (implicit)."""
    def __init__(self, factors: int = 64, lr: float = 0.05, reg: float = 0.002,
                 epochs: int = 20, neg_ratio: int = 4, seed: int = 42):
        self.f = factors; self.lr = lr; self.reg = reg
        self.epochs = epochs; self.neg_ratio = neg_ratio; self.seed = seed
        self.P = None; self.Q = None

    def fit(self, R: csr_matrix):
        R = R.tocsr().astype(float)
        U, I = R.shape
        rng = np.random.default_rng(self.seed)
        P = 0.01 * rng.normal(size=(U, self.f))
        Q = 0.01 * rng.normal(size=(I, self.f))

        users = np.arange(U, dtype=int)
        pos_lists = [R.indices[R.indptr[u]:R.indptr[u+1]] for u in users]
        pos_sets = [set(a.tolist()) for a in pos_lists]

        logger.info(f"[model:bpr] start train U={U} I={I} f={self.f} epochs={self.epochs} lr={self.lr} reg={self.reg}")

        for ep in range(1, self.epochs + 1):
            rng.shuffle(users)
            updates = 0
            for u in users:
                pos_items = pos_lists[u]
                if pos_items.size == 0: continue
                for i in rng.choice(pos_items, size=min(pos_items.size, self.neg_ratio), replace=True):
                    j = int(rng.integers(0, I))
                    while j in pos_sets[u]:
                        j = int(rng.integers(0, I))
                    x_uij = (P[u] @ (Q[i] - Q[j]))
                    sig = 1.0 / (1.0 + np.exp(-x_uij))
                    grad_u = (1 - sig) * (Q[i] - Q[j]) - self.reg * P[u]
                    grad_i = (1 - sig) * P[u] - self.reg * Q[i]
                    grad_j = -(1 - sig) * P[u] - self.reg * Q[j]
                    P[u] += self.lr * grad_u
                    Q[i] += self.lr * grad_i
                    Q[j] += self.lr * grad_j
                    updates += 1
            logger.info(f"[model:bpr] epoch={ep}/{self.epochs} updates={updates} ||P||={np.linalg.norm(P):.3e} ||Q||={np.linalg.norm(Q):.3e}")

        self.P, self.Q = P, Q
        logger.info("[model:bpr] training done")
        return self

    def score_user(self, u: int, R: csr_matrix, exclude_seen: bool = True) -> np.ndarray:
        assert self.P is not None and self.Q is not None
        scores = self.P[u] @ self.Q.T
        if exclude_seen:
            s, e = R.indptr[u], R.indptr[u+1]
            scores[R.indices[s:e]] = -np.inf
        return scores
