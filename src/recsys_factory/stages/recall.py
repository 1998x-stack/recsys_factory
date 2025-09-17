from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any
from scipy.sparse import csr_matrix
from ..factory import register, get
from ..log import logger

@register("stage", "recall")
class RecallStage:
    """Use a 'model' to produce candidate pool per user."""
    def __init__(self, model: str, params: Dict[str, Any], topk: int = 300):
        Model = get("model", model)
        self.model = Model(**params)
        self.topk = topk

    def fit(self, R_train: csr_matrix):
        t0 = time.time()
        self.model.fit(R_train)
        logger.info(f"[stage:recall] model={type(self.model).__name__} fit_done in {time.time()-t0:.2f}s")
        return self

    def run(self, R_train: csr_matrix, users: np.ndarray) -> Dict[int, np.ndarray]:
        t0 = time.time()
        out = {}
        for u in users:
            scores = self.model.score_user(u, R_train, exclude_seen=True)
            k = min(self.topk, scores.size)
            cand = np.argpartition(-scores, k-1)[:k]
            out[u] = cand
        logger.info(f"[stage:recall] topk={self.topk} users={len(users)} time={time.time()-t0:.2f}s")
        return out
