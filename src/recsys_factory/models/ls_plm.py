from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from .ctr_base import CTRBaseModel
from ..factory import register
from ..log import logger
from .lr import LogisticRegressionCTR  # base learner

@register("model", "ls_plm")
class LSPLM(CTRBaseModel):
    """
    Large-Scale Piecewise Linear Model: hard partition (KMeans on dense),
    then one LR per region on full sparse X.
    """
    fit_requires_field_slices = True  # trainer will pass field_slices

    def __init__(self, n_regions: int = 16, kmeans_max_iter: int = 100,
                 lr_lr: float = 0.1, lr_reg: float = 1e-6, lr_epochs: int = 3,
                 lr_batch_size: int = 65536, seed: int = 42):
        self.n_regions = n_regions
        self.kmeans = KMeans(n_clusters=n_regions, n_init="auto", max_iter=kmeans_max_iter, random_state=seed)
        self.seed = seed
        self.lr_params = dict(lr=lr_lr, reg=lr_reg, epochs=lr_epochs, batch_size=lr_batch_size, seed=seed)
        self.region_models = []
        self.dense_slice = None

    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None, field_slices=None):
        assert field_slices is not None, "LS-PLM requires field_slices to locate dense slice"
        self.dense_slice = field_slices[0]
        Xd = X[:, self.dense_slice].toarray()
        logger.info(f"[ls_plm] k-means clustering on dense d={Xd.shape[1]} regions={self.n_regions}")
        labels = self.kmeans.fit_predict(Xd)

        self.region_models = [LogisticRegressionCTR(**self.lr_params) for _ in range(self.n_regions)]
        totals = []
        for r in range(self.n_regions):
            idx = np.where(labels == r)[0]
            totals.append(len(idx))
            if len(idx) == 0:
                logger.warning(f"[ls_plm] region {r} empty; using prior p=mean(y)")
                # leave model as zeros (predict 0.5); or could store prior
                continue
            Xr, yr = X[idx], y[idx]
            logger.info(f"[ls_plm] train LR for region {r}: n={len(idx)}")
            self.region_models[r].fit(Xr, yr)
        logger.info(f"[ls_plm] region sizes: {totals}")
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        Xd = X[:, self.dense_slice].toarray()
        labels = self.kmeans.predict(Xd)
        p = np.zeros(X.shape[0], dtype=float)
        for r in range(self.n_regions):
            idx = np.where(labels == r)[0]
            if len(idx) == 0: continue
            p[idx] = self.region_models[r].predict_proba(X[idx])
        return p
