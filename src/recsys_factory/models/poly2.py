from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .ctr_base import CTRBaseModel
from ..features.poly import hashed_poly2
from ..factory import register
from ..log import logger

@register("model", "poly2_lr")
class Poly2LogReg(CTRBaseModel):
    """Explicit 2nd-order hashed interactions + LR."""
    def __init__(self, out_bins: int = 500_000, lr: float = 0.1, reg: float = 1e-6, epochs: int = 3, batch_size: int = 65536, seed: int = 42):
        self.out_bins = out_bins; self.lr = lr; self.reg = reg; self.epochs = epochs; self.bs = batch_size; self.seed = seed
        self.lr_core = None; self.X_shape = None

    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None):
        X2 = hashed_poly2(X, self.out_bins, seed=self.seed)
        self.X_shape = X2.shape[1]
        from .lr import LogisticRegressionCTR
        self.lr_core = LogisticRegressionCTR(lr=self.lr, reg=self.reg, epochs=self.epochs, batch_size=self.bs, seed=self.seed)
        self.lr_core.fit(X2, y, hashed_poly2(X_val, self.out_bins, seed=self.seed) if X_val is not None else None, y_val)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        X2 = hashed_poly2(X, self.out_bins, seed=self.seed)
        return self.lr_core.predict_proba(X2)
