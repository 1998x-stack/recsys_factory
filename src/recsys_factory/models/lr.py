from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .ctr_base import CTRBaseModel
from ..factory import register
from ..log import logger

@register("model", "lr")
class LogisticRegressionCTR(CTRBaseModel):
    """Sparse LR with SGD (logloss)."""
    def __init__(self, lr: float = 0.1, reg: float = 1e-6, epochs: int = 3, batch_size: int = 65536, seed: int = 42):
        self.lr = lr; self.reg = reg; self.epochs = epochs; self.bs = batch_size; self.seed = seed
        self.w = None

    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(d)
        logger.info(f"[lr] n={n} d={d} epochs={self.epochs} bs={self.bs} lr={self.lr} reg={self.reg}")
        for ep in range(1, self.epochs+1):
            idx = rng.permutation(n)
            losses = []
            for s in range(0, n, self.bs):
                batch = idx[s:s+self.bs]
                Xb = X[batch]; yb = y[batch]
                z = Xb @ self.w
                p = 1/(1+np.exp(-z))
                g = Xb.T @ (p - yb) / len(batch) + self.reg * self.w
                self.w -= self.lr * g.A.ravel()
                loss = float(-(yb*np.log(p+1e-15)+(1-yb)*np.log(1-p+1e-15)).mean())
                losses.append(loss)
            msg = f"[lr] epoch={ep} train_logloss={np.mean(losses):.5f}"
            if X_val is not None:
                pv = self.predict_proba(X_val); 
                from ..metrics.classification import logloss, roc_auc
                msg += f" valid_logloss={logloss(y_val,pv):.5f} AUC={roc_auc(y_val,pv):.4f}"
            logger.info(msg)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        z = X @ self.w
        return 1/(1+np.exp(-z))
