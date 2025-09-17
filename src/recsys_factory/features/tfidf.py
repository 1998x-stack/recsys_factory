from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, diags
from ..log import logger

def _col_nnz(X: csr_matrix) -> np.ndarray:
    csc = X.tocsc()
    return np.diff(csc.indptr)  # non-zeros per column

def _row_nnz(X: csr_matrix) -> np.ndarray:
    csr = X.tocsr()
    return np.diff(csr.indptr)  # non-zeros per row

def item_tfidf(R: csr_matrix) -> csr_matrix:
    """Column-wise TF-IDF (common for ItemCF)."""
    df = _col_nnz(R)
    idf = np.log((R.shape[0]) / np.maximum(1, df))
    W = diags(idf)
    X = R.tocsr().astype(float) @ W
    logger.debug("[feature] item_tfidf applied")
    return X

def user_tfidf(R: csr_matrix) -> csr_matrix:
    """Row-wise TF-IDF (symmetric idea)."""
    df = _row_nnz(R)
    idf = np.log((R.shape[1]) / np.maximum(1, df))
    D = diags(idf)
    X = D @ R.tocsr().astype(float)
    logger.debug("[feature] user_tfidf applied")
    return X
