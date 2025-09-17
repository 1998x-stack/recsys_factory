from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .base import InteractionDataset
from ..factory import register
from ..log import logger

@register("dataset", "movielens100k")
class MovieLensExplicit:
    """Load MovieLens-100K 'u.data' (user item rating timestamp)."""
    def __init__(self, path: str):
        logger.info(f"[dataset] loading MovieLens from {path}")
        data = np.loadtxt(path, dtype=int)
        users_raw = data[:, 0].astype(str)
        items_raw = data[:, 1].astype(str)
        ratings = data[:, 2].astype(float)
        ts = data[:, 3].astype(int)

        u_ids = {u: i for i, u in enumerate(np.unique(users_raw))}
        i_ids = {i: j for j, i in enumerate(np.unique(items_raw))}
        rows = np.array([u_ids[u] for u in users_raw], dtype=int)
        cols = np.array([i_ids[v] for v in items_raw], dtype=int)

        R = csr_matrix((ratings, (rows, cols)), shape=(len(u_ids), len(i_ids)))
        T = csr_matrix((ts, (rows, cols)), shape=R.shape)
        logger.info(f"[dataset] users={len(u_ids)} items={len(i_ids)} nnz={R.nnz}")
        self.ds = InteractionDataset(u_ids, i_ids, R, T)

    def materialize(self) -> InteractionDataset:
        return self.ds
