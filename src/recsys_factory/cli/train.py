from __future__ import annotations
import argparse
from ..config import load_config
from ..factory import get
from ..log import logger

# CF deps
from ..data import MovieLensExplicit, leave_one_out_split, time_order_split
from ..trainer import OfflineTrainer

# CTR deps
from ..data.ctr import ToyCTR, CriteoCSV
from ..trainer.ctr_offline import CTROfflineTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger.info(f"[cli] task={cfg.task}")

    if cfg.task == "cf":
        logger.info(f"[cli] stages={[s.name+':'+s.component for s in cfg.stages]} metrics={cfg.metrics}")
        DS = get("dataset", cfg.dataset["name"])
        ds = DS(**{k: v for k, v in cfg.dataset.items() if k != "name"}).materialize()
        logger.info(f"[cli] dataset users={ds.n_users} items={ds.n_items} nnz={ds.R.nnz}")
        if cfg.split["method"] == "loo":
            R_train, R_test = leave_one_out_split(ds.R, seed=cfg.split.get("seed", 42))
        else:
            R_train, R_test = time_order_split(ds.R, ds.timestamps, cfg.split.get("ratio", 0.2))
        Trainer = get("trainer", cfg.trainer["name"])
        tr = Trainer()
        metrics, _ = tr.run(cfg, ds, R_train, R_test)
        logger.info(f"[cli] FINAL {metrics}")
    else:
        logger.info(f"[cli] model={cfg.model['name']} metrics={cfg.metrics}")
        DS = get("dataset", cfg.dataset["name"])
        ds = DS(**{k: v for k, v in cfg.dataset.items() if k != "name"})
        ds = ds.materialize(cfg.split)
        Trainer = get("trainer", cfg.trainer["name"])
        tr = Trainer()
        metrics, _ = tr.run(cfg, ds)
        logger.info(f"[cli] FINAL {metrics}")

if __name__ == "__main__":
    main()
