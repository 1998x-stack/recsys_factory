from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any
from scipy.sparse import csr_matrix
from ..factory import register, get
from ..log import logger

@register("stage", "rank")
class RankStage:
    """Score candidates with another 'model' and take topk."""
    def __init__(self, model: str, params: Dict[str, Any], topk: int = 50):
        Model = get("model", model)
        self.model = Model(**params)
        self.topk = topk

    def fit(self, R_train: csr_matrix):
        t0 = time.time()
        self.model.fit(R_train)
        logger.info(f"[stage:rank] model={type(self.model).__name__} fit_done in {time.time()-t0:.2f}s")
        return self

    def run(self, R_train: csr_matrix, users: np.ndarray, candidates: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        t0 = time.time()
        ranked = {}
        for u in users:
            scores = self.model.score_user(u, R_train, exclude_seen=True)
            cand = candidates.get(u, np.arange(R_train.shape[1]))
            s = scores[cand]
            order = np.argsort(-s)[: min(self.topk, s.size)]
            ranked[u] = cand[order]
        logger.info(f"[stage:rank] topk={self.topk} users={len(users)} time={time.time()-t0:.2f}s")
        return ranked
