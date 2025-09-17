from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import GradientBoostingClassifier
from .ctr_base import CTRBaseModel
from ..factory import register
from ..log import logger
from ..features.gbdt_leaf import leaves_to_csr
from .lr import LogisticRegressionCTR  # reuse our sparse LR

@register("model", "gbdt_lr")
class GBDTPlusLR(CTRBaseModel):
    """
    Train GBDT on dense features -> transform to leaf one-hots -> concat with raw sparse X -> LR.
    Params mirror classic GBDT+LR baseline; dense slice must be first field in dataset.field_slices.
    """
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 subsample: float = 1.0,
                 include_raw: bool = True,
                 lr_lr: float = 0.1,
                 lr_reg: float = 1e-6,
                 lr_epochs: int = 3,
                 lr_batch_size: int = 65536,
                 seed: int = 42):
        self.gbdt = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample, random_state=seed
        )
        self.include_raw = include_raw
        self.lr = LogisticRegressionCTR(lr=lr_lr, reg=lr_reg, epochs=lr_epochs,
                                        batch_size=lr_batch_size, seed=seed)
        self.dense_slice = None
        self._leaf_offsets = None

    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None, field_slices=None):
        assert field_slices is not None, "GBDT+LR requires field_slices to locate dense slice"
        self.dense_slice = field_slices[0]  # convention: first slice is dense block
        Xd = X[:, self.dense_slice].toarray()
        logger.info(f"[gbdt_lr] train GBDT on dense d={Xd.shape[1]} trees={self.gbdt.n_estimators} depth={self.gbdt.max_depth}")
        self.gbdt.fit(Xd, y)
        train_leaf = self.gbdt.apply(Xd).astype(int)
        X_leaf, offsets = leaves_to_csr(train_leaf)
        self._leaf_offsets = offsets

        X_lr = hstack([X, X_leaf], format="csr") if self.include_raw else X_leaf
        logger.info(f"[gbdt_lr] LR input dim={X_lr.shape[1]} (include_raw={self.include_raw})")
        Xv_lr = None
        if X_val is not None:
            Xdv = X_val[:, self.dense_slice].toarray()
            val_leaf = self.gbdt.apply(Xdv).astype(int)
            X_leaf_v, _ = leaves_to_csr(val_leaf)
            Xv_lr = hstack([X_val, X_leaf_v], format="csr") if self.include_raw else X_leaf_v
        self.lr.fit(X_lr, y, Xv_lr, y_val)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        Xd = X[:, self.dense_slice].toarray()
        leaf = self.gbdt.apply(Xd).astype(int)
        X_leaf, _ = leaves_to_csr(leaf)
        X_lr = hstack([X, X_leaf], format="csr") if self.include_raw else X_leaf
        return self.lr.predict_proba(X_lr)
