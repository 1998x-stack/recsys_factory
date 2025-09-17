from __future__ import annotations
import numpy as np, pandas as pd
from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix, hstack
from ..log import logger

def _hash_str(s: str, mod: int) -> int:
    return (hash(s) % mod + mod) % mod

def build_design_matrix(
    df: pd.DataFrame,
    dense_cols: List[str],
    cat_cols: List[str],
    freq_threshold: int = 2,
    hash_bins: int = 200_000,
) -> Tuple[csr_matrix, List[slice], Dict]:
    """Return X (csr), field_slices (for FFM), meta."""
    mats = []
    field_slices = []
    start = 0

    # dense
    if dense_cols:
        dense = df[dense_cols].astype(float).fillna(0.0).values
        mats.append(csr_matrix(dense))
        field_slices.append(slice(start, start + dense.shape[1])); start += dense.shape[1]
        logger.info(f"[fe] dense_cols={len(dense_cols)}")

    # categorical hashed one-hot with frequency cutoff
    for col in cat_cols:
        vc = df[col].astype(str).value_counts()
        keep = set(vc[vc >= freq_threshold].index)
        xs = df[col].astype(str).apply(lambda x: x if x in keep else "__OTHER__")
        rows = np.arange(len(df), dtype=int)
        cols_idx = np.array([_hash_str(f"{col}={v}", hash_bins) for v in xs], dtype=int)
        data = np.ones(len(df), dtype=float)
        mat = csr_matrix((data, (rows, cols_idx)), shape=(len(df), hash_bins))
        mats.append(mat)
        field_slices.append(slice(start, start + hash_bins)); start += hash_bins
        logger.info(f"[fe] cat={col} kept={len(keep)} bins={hash_bins}")

    X = hstack(mats, format="csr")
    meta = {"dense_cols": dense_cols, "cat_cols": cat_cols, "hash_bins": hash_bins}
    logger.info(f"[fe] X shape={X.shape}, fields={len(field_slices)}")
    return X, field_slices, meta
