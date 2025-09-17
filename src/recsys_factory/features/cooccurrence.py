from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from ..log import logger

def cooc_counts(R: csr_matrix) -> csr_matrix:
    B = R.copy()
    B.data = np.ones_like(B.data)
    S = (B.T @ B).tocsr()
    logger.debug("[feature] co-occurrence computed (R^T R)")
    return S
