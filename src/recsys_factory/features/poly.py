from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from ..log import logger

def hashed_poly2(X: csr_matrix, out_bins: int, seed: int = 42) -> csr_matrix:
    """Explicit 2nd-order interactions via hashing trick: concat original X + hashed cross."""
    rng = np.random.default_rng(seed)
    X = X.tocsr().astype(float)
    n, d = X.shape
    rows, cols, data = [], [], []

    # keep original features
    X_coo = X.tocoo()
    rows.extend(X_coo.row.tolist())
    cols.extend(X_coo.col.tolist())
    data.extend(X_coo.data.tolist())

    # pairwise cross for nnz per row (sparse-safe)
    indptr = X.indptr; indices = X.indices; values = X.data
    base = d
    for i in range(n):
        s, e = indptr[i], indptr[i+1]
        idx = indices[s:e]; val = values[s:e]
        m = len(idx)
        for a in range(m):
            for b in range(a+1, m):
                h = (idx[a] * 1315423911 + idx[b] * 2654435761) & 0x7fffffff
                col = base + (h % out_bins)
                rows.append(i); cols.append(col); data.append(val[a] * val[b])

    X2 = csr_matrix((data, (rows, cols)), shape=(n, d + out_bins))
    logger.info(f"[poly2] expanded from d={d} to d'={X2.shape[1]}")
    return X2
