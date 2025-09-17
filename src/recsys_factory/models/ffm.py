from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from .ctr_base import CTRBaseModel
from ..factory import register
from ..log import logger

@register("model", "ffm")
class FieldAwareFM(CTRBaseModel):
    """Simplified FFM with SGD (needs field_slices from dataset)."""
    def __init__(self, factors: int = 8, lr: float = 0.05, reg: float = 1e-6,
                 epochs: int = 3, batch_size: int = 65536, seed: int = 42):
        self.k=factors; self.lr=lr; self.reg=reg; self.epochs=epochs; self.bs=batch_size; self.seed=seed
        self.W = None  # [d, F, k]
        self.w0=0.0; self.w=None
        self.field_slices=None

    def _score(self, X: csr_matrix) -> np.ndarray:
        # x is sparse row; FFM: sum_{i<j} <V_{i,f_j}, V_{j,f_i}> x_i x_j + linear + w0
        n, d = X.shape
        F = len(self.field_slices)
        out = np.full(n, self.w0, dtype=float)
        if self.w is not None:
            out += (X @ self.w).A.ravel()
        # pairwise interactions
        indptr, indices, data = X.indptr, X.indices, X.data
        for r in range(n):
            s, e = indptr[r], indptr[r+1]
            idx = indices[s:e]; val = data[s:e]
            # fields
            f_idx = [self._field_of(i) for i in idx]
            m = len(idx)
            ssum = 0.0
            for a in range(m):
                fa = f_idx[a]
                for b in range(a+1, m):
                    fb = f_idx[b]
                    va = self.W[idx[a], fb]    # [k]
                    vb = self.W[idx[b], fa]    # [k]
                    ssum += np.dot(va, vb) * val[a] * val[b]
            out[r] += ssum
        return out

    def _field_of(self, col: int) -> int:
        for f, sl in enumerate(self.field_slices):
            if sl.start <= col < sl.stop: return f
        return len(self.field_slices) - 1

    def fit(self, X: csr_matrix, y: np.ndarray, X_val=None, y_val=None, field_slices=None):
        n, d = X.shape; rng = np.random.default_rng(self.seed)
        if field_slices is None:
            raise ValueError("FFM requires field_slices from dataset.")
        self.field_slices = field_slices
        F = len(field_slices)
        self.w = np.zeros(d)
        self.W = 0.01 * rng.normal(size=(d, F, self.k))
        logger.info(f"[ffm] n={n} d={d} F={F} k={self.k} epochs={self.epochs} bs={self.bs}")
        for ep in range(1, self.epochs+1):
            idx = rng.permutation(n); losses=[]
            for s in range(0, n, self.bs):
                batch = idx[s:s+self.bs]
                Xb = X[batch]; yb=y[batch]
                z = self._score(Xb)
                p = 1/(1+np.exp(-z))
                g = (p - yb)
                # update w0/w
                self.w0 -= self.lr * g.mean()
                self.w *= (1 - self.lr * self.reg)
                self.w -= self.lr * (Xb.T @ (g / len(batch))).A.ravel()
                # update W (very simplified; per-nz pair loops)
                indptr, indices, data = Xb.indptr, Xb.indices, Xb.data
                for r in range(Xb.shape[0]):
                    s0, e0 = indptr[r], indptr[r+1]
                    idxr = indices[s0:e0]; valr = data[s0:e0]
                    fr = [self._field_of(i) for i in idxr]
                    for a in range(len(idxr)):
                        fa = fr[a]
                        for b in range(a+1, len(idxr)):
                            fb = fr[b]
                            grad = g[r] * valr[a] * valr[b]
                            self.W[idxr[a], fb] *= (1 - self.lr*self.reg)
                            self.W[idxr[b], fa] *= (1 - self.lr*self.reg)
                            tmp_a = self.W[idxr[a], fb].copy()
                            self.W[idxr[a], fb] -= self.lr * grad * self.W[idxr[b], fa]
                            self.W[idxr[b], fa] -= self.lr * grad * tmp_a
                loss = float(-(yb*np.log(p+1e-15)+(1-yb)*np.log(1-p+1e-15)).mean())
                losses.append(loss)
            msg = f"[ffm] epoch={ep} train_logloss={np.mean(losses):.5f}"
            if X_val is not None:
                from ..metrics.classification import logloss, roc_auc
                pv = self.predict_proba(X_val)
                msg += f" valid_logloss={logloss(y_val,pv):.5f} AUC={roc_auc(y_val,pv):.4f}"
            logger.info(msg)
        return self

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        z = self._score(X)
        return 1/(1+np.exp(-z))
