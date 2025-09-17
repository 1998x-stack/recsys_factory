from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .base import BaseModel
from ..factory import register
from ..log import logger

@register("model", "als")
class ALS(BaseModel):
    """Implicit ALS (Hu et al. 2008) — small/medium reference implementation."""
    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 40.0, iters: int = 10, seed: int = 123):
        self.f = factors; self.reg = reg; self.alpha = alpha; self.iters = iters; self.seed = seed
        self.U = None; self.V = None

    def fit(self, R: csr_matrix):
        R = R.tocsr().astype(float)
        U, I = R.shape
        rng = np.random.default_rng(self.seed)
        Ue = rng.normal(0, 0.01, size=(U, self.f))
        Ve = rng.normal(0, 0.01, size=(I, self.f))
        I_f = np.eye(self.f)
        logger.info(f"[model:als] start train U={U} I={I} f={self.f} iters={self.iters} reg={self.reg} alpha={self.alpha}")

        for it in range(1, self.iters + 1):
            # fix V, solve U
            YtY = Ve.T @ Ve + self.reg * I_f
            for u in range(U):
                row = R.getrow(u)
                idx = row.indices
                if idx.size == 0: continue
                Cu = (1.0 + self.alpha * row.data)          # confidences
                Y_u = Ve[idx]                                # [nu, f]
                A = YtY + (Y_u.T * (Cu - 1.0)) @ Y_u
                b = (Y_u * Cu[:, None]).sum(axis=0)
                Ue[u] = np.linalg.solve(A, b)

            # fix U, solve V
            XtX = Ue.T @ Ue + self.reg * I_f
            for i in range(I):
                col = R.getcol(i).tocsr()
                idx = col.indices
                if idx.size == 0: continue
                Ci = (1.0 + self.alpha * col.data)
                X_i = Ue[idx]
                A = XtX + (X_i.T * (Ci - 1.0)) @ X_i
                b = (X_i * Ci[:, None]).sum(axis=0)
                Ve[i] = np.linalg.solve(A, b)

            frob = np.linalg.norm(Ue) * np.linalg.norm(Ve)
            logger.info(f"[model:als] iter={it}/{self.iters} ||U||*||V||≈{frob:.3e}")

        self.U, self.V = Ue, Ve
        logger.info("[model:als] training done")
        return self

    def score_user(self, u: int, R: csr_matrix, exclude_seen: bool = True) -> np.ndarray:
        assert self.U is not None and self.V is not None
        scores = self.U[u] @ self.V.T
        if exclude_seen:
            s, e = R.indptr[u], R.indptr[u+1]
            scores[R.indices[s:e]] = -np.inf
        return scores
