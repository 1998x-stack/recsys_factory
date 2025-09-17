from __future__ import annotations
from ..factory import register, get
from ..log import logger
from ..metrics.classification import logloss, roc_auc
from ..data.ctr_base import CTRDataset

@register("trainer", "ctr_offline")
class CTROfflineTrainer:
    def __init__(self): ...

    def run(self, cfg, dataset: CTRDataset, *args, **kwargs):
        mname = cfg.model["name"]
        params = cfg.model.get("params", {})
        Model = get("model", mname)
        model = Model(**params)
        logger.info(f"[trainer:ctr] model={mname} params={params}")

        fit_kwargs = {}
        if getattr(model, "fit_requires_field_slices", False):
            fit_kwargs["field_slices"] = dataset.field_slices

        model.fit(dataset.X_train, dataset.y_train,
                  dataset.X_valid, dataset.y_valid, **fit_kwargs)

        pv = model.predict_proba(dataset.X_valid)
        metrics = {"AUC": roc_auc(dataset.y_valid, pv), "LogLoss": logloss(dataset.y_valid, pv)}
        logger.info(f"[trainer:ctr] FINAL {metrics}")
        return metrics, {"proba": pv}
