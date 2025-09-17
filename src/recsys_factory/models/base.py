from __future__ import annotations
from typing import Iterable
import numpy as np
from scipy.sparse import csr_matrix

class BaseModel:
    def fit(self, R: csr_matrix) -> "BaseModel": raise NotImplementedError
    def score_user(self, u: int, R: csr_matrix, exclude_seen: bool = True) -> np.ndarray:
        raise NotImplementedError
    def score_users(self, users: Iterable[int], R: csr_matrix, exclude_seen: bool = True) -> np.ndarray:
        return np.vstack([self.score_user(u, R, exclude_seen) for u in users])
