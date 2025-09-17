from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .ctr_base import CTRBaseModel
from ..factory import register
from ..log import logger

@register("model", "fm")
class FactorizationMachines(CTRBaseModel):
    """Binary FM with SGD."""
    def __init__(self, factors: int = 16, lr: float = 0.05, reg_w: float = 1e-6, reg_v: float = 1e-6,
                 epochs: int = 3, batch_size: int = 65536, seed: int = 42):
        self.k = factors; self.lr=lr; self.reg_w=reg_w; self.reg_v=reg_v
        self.epochs=epochs; self.bs=batch_size; self.seed=seed
        self.w0=0.0; self.w=None; self.V=None

    def _score_batch(self, Xb: csr_matrix) -> np.ndarray:
        # FM: w0 + Xw + 0.5*( (X V)^2 - (X^2)(V^2) )
        linear = self.w0 + Xb @ self.w
        XV = Xb @ self.V
        X2 = Xb.copy(); X2.data **= 2.0
        V2 = self.V**2
        inter = 0.5 * (XV**2 - (X2 @ V2)).sum(axis=1).A.ravel()
        return linear.A.ravel() + inter if hasattr(linear, "A") else linear + inter

    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None):
        n, d = X.shape; rng = np.random.default_rng(self.seed)
        self.w = np.zeros(d); self.V = 0.01 * rng.normal(size=(d, self.k))
        logger.info(f"[fm] n={n} d={d} k={self.k} epochs={self.epochs} bs={self.bs}")
        for ep in range(1, self.epochs+1):
            idx = rng.permutation(n); losses=[]
            for s in range(0, n, self.bs):
                batch = idx[s:s+self.bs]
                Xb = X[batch]; yb=y[batch]
                z = self._score_batch(Xb)
                p = 1/(1+np.exp(-z))
                # grads
                g = (p - yb)  # [bs]
                # update w0
                self.w0 -= self.lr * g.mean()
                # update w (sparse)
                self.w *= (1 - self.lr * self.reg_w)
                self.w -= self.lr * (Xb.T @ (g / len(batch))).A.ravel()
                # update V
                XV = Xb @ self.V                          # [bs, k]
                for f in range(self.k):
                    # gradient wrt V[:,f]: X^T (g * (X V_f - X .* V_{:,f}))
                    t1 = Xb.T @ (g * XV[:, f] / len(batch))
                    X_col_sq = Xb.copy(); X_col_sq.data *= Xb.data  # misuse; but below we use diag trick
                    t2 = (Xb.T.multiply(self.V[:, f])).sum(axis=1)  # approx term
                    grad = t1.A.ravel() - t2.A.ravel() / len(batch)
                    self.V[:, f] *= (1 - self.lr * self.reg_v)
                    self.V[:, f] -= self.lr * grad
                loss = float(-(yb*np.log(p+1e-15)+(1-yb)*np.log(1-p+1e-15)).mean())
                losses.append(loss)
            msg = f"[fm] epoch={ep} train_logloss={np.mean(losses):.5f}"
            if X_val is not None:
                from ..metrics.classification import logloss, roc_auc
                pv = self.predict_proba(X_val)
                msg += f" valid_logloss={logloss(y_val,pv):.5f} AUC={roc_auc(y_val,pv):.4f}"
            logger.info(msg)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        z = self._score_batch(X)
        return 1/(1+np.exp(-z))
