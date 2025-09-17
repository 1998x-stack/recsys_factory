from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.sparse import csr_matrix

@dataclass
class CTRDataset:
    X_train: csr_matrix
    y_train: np.ndarray
    X_valid: csr_matrix
    y_valid: np.ndarray
    field_slices: Optional[List[slice]] = None  # FFM 用：每个 field 的特征切片
    meta: Dict[str, any] = None
