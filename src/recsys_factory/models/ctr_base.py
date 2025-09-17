from __future__ import annotations
from scipy.sparse import csr_matrix
import numpy as np

class CTRBaseModel:
    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None): ...
    def predict_proba(self, X: csr_matrix) -> np.ndarray: ...
