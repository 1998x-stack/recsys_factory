from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from ..log import logger

def leave_one_out_split(R: csr_matrix, seed: int = 42):
    """LOO per user -> test one positive (binary)."""
    logger.info("[split] method=LOO")
    rng = np.random.default_rng(seed)
    R = R.tocsr().astype(float)
    U, _ = R.shape
    train = R.copy()
    test = csr_matrix(R.shape, dtype=float)
    moved = 0
    for u in range(U):
        s, e = R.indptr[u], R.indptr[u+1]
        items = R.indices[s:e]
        if items.size < 2: continue
        pick = int(rng.choice(items))
        train[u, pick] = 0.0
        test[u, pick] = 1.0
        moved += 1
    train.eliminate_zeros(); test.eliminate_zeros()
    logger.info(f"[split] moved={moved} train_nnz={train.nnz} test_nnz={test.nnz}")
    return train, test

def time_order_split(R: csr_matrix, T: csr_matrix, holdout_ratio: float = 0.2):
    logger.info(f"[split] method=time ratio={holdout_ratio}")
    R = R.tocsr().astype(float)
    T = T.tocsr().astype(int)
    U, _ = R.shape
    train = R.copy()
    test = csr_matrix(R.shape, dtype=float)
    moved = 0
    for u in range(U):
        s, e = R.indptr[u], R.indptr[u+1]
        items = R.indices[s:e]
        if items.size == 0: continue
        times = np.array([T[u, i] for i in items])
        order = np.argsort(times)
        k = max(1, int(items.size * holdout_ratio))
        test_items = items[order[-k:]]
        for it in test_items:
            train[u, it] = 0.0
            test[u, it] = 1.0
            moved += 1
    train.eliminate_zeros(); test.eliminate_zeros()
    logger.info(f"[split] moved={moved} train_nnz={train.nnz} test_nnz={test.nnz}")
    return train, test
