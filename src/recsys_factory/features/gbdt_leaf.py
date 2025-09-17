from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, hstack
from typing import List, Tuple
from loguru import logger

def leaves_to_csr(leaf_idx: np.ndarray) -> Tuple[csr_matrix, List[int]]:
    """
    Convert sklearn GBDT .apply(...) output -> CSR one-hot per tree.
    leaf_idx: shape [n_samples, n_trees], each value is node index of a leaf in that tree.
    Return:
      X_leaf: CSR [n, sum_leaves], one-hot
      offsets_per_tree: list of feature offsets for each tree
    """
    n, T = leaf_idx.shape
    cols, rows, data, offsets = [], [], [], []
    offset = 0
    for t in range(T):
        uniq = np.unique(leaf_idx[:, t])
        mapping = {leaf_id: j for j, leaf_id in enumerate(uniq)}
        offsets.append(offset)
        for i in range(n):
            j = mapping[leaf_idx[i, t]]
            rows.append(i); cols.append(offset + j); data.append(1.0)
        offset += len(uniq)
    X_leaf = csr_matrix((data, (rows, cols)), shape=(n, offset))
    logger.info(f"[gbdt_leaf] trees={T} total_leaf_feats={offset}")
    return X_leaf, offsets
