from __future__ import annotations
import time
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict
from ..factory import get, register
from ..metrics import precision_at_k, recall_at_k, ndcg_at_k, map_at_k
from ..log import logger

_METRIC_FUN = {
    "P": precision_at_k,
    "R": recall_at_k,
    "NDCG": ndcg_at_k,
    "MAP": map_at_k,
}

@register("trainer", "offline")
class OfflineTrainer:
    """Orchestrate stages and evaluate on a holdout test set."""
    def __init__(self): ...

    def run(self, cfg, dataset, train_R: csr_matrix, test_R: csr_matrix):
        t0 = time.time()
        users = np.arange(train_R.shape[0], dtype=int)
        logger.info(f"[trainer] users={len(users)} items={train_R.shape[1]} train_nnz={train_R.nnz} test_nnz={test_R.nnz}")

        # build stages from config
        stage_objs = []
        for s in cfg.stages:
            Stage = get("stage", s.name)
            kwargs = dict(s.params)
            if s.name in ("recall", "rank"):
                stage = Stage(model=s.component, params=kwargs, topk=s.topk or 100)
            else:
                stage = Stage(**kwargs)  # e.g., rerank_mmr
            stage.fit(train_R)
            stage_objs.append((s.name, stage, s.topk))

        # run pipeline
        outputs = {}
        last = None
        for name, stage, _ in stage_objs:
            tic = time.time()
            if name == "recall":
                last = stage.run(train_R, users)
            elif name == "rank":
                last = stage.run(train_R, users, last)
            else:  # rerank
                last = stage.run(train_R, users, last)
            outputs[name] = last
            logger.info(f"[trainer] stage={name} done in {time.time()-tic:.2f}s")

        # evaluate on test positives
        metrics = {m: 0.0 for m in cfg.metrics}
        denom = 0
        final = last
        for u in users:
            truth = test_R.getrow(u).indices
            if truth.size == 0: continue
            pred = final[u]
            for m in cfg.metrics:
                name, k = m.split("@"); k = int(k)
                metrics[m] += _METRIC_FUN[name](pred, truth, k)
            denom += 1
        for m in metrics:
            metrics[m] = metrics[m] / max(1, denom)
        logger.info(f"[trainer] eval_users={denom} metrics={metrics} total_time={time.time()-t0:.2f}s")
        return metrics, outputs
