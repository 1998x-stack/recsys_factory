from __future__ import annotations
import time
import numpy as np
from typing import Dict
from scipy.sparse import csr_matrix
from ..factory import register
from ..log import logger

@register("stage", "rerank_mmr")
class MMRRerankStage:
    """MMR diversity re-ranker using item co-occurrence similarity as diversity penalty."""
    def __init__(self, lambda_: float = 0.7):
        self.lmb = lambda_
        self.S = None

    def fit(self, R_train: csr_matrix):
        B = R_train.copy(); B.data[:] = 1.0
        self.S = (B.T @ B).tocsr()
        logger.info("[stage:rerank_mmr] built item co-occurrence similarity")
        return self

    def run(self, R_train: csr_matrix, users: np.ndarray, ranked: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        t0 = time.time()
        out = {}
        S = self.S
        for u in users:
            cand = ranked[u]
            if cand.size <= 2:
                out[u] = cand; continue
            selected = [cand[0]]
            rest = list(cand[1:])
            rel = np.linspace(1.0, 0.0, num=len(cand)-1)  # proxy relevance
            while rest:
                sims = np.array([max((float(S[it, j]) for j in selected), default=0.0) for it in rest])
                mmr = self.lmb * rel[:len(rest)] - (1 - self.lmb) * sims
                pick = int(np.argmax(mmr))
                selected.append(rest.pop(pick))
            out[u] = np.array(selected, dtype=int)
        logger.info(f"[stage:rerank_mmr] users={len(users)} time={time.time()-t0:.2f}s")
        return out
