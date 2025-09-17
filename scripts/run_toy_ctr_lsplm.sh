#!/usr/bin/env bash
set -euo pipefail
python scripts/gen_toy_ctr.py
echo "[run] CTR: LS-PLM on toy_ctr"
python -m recsys_factory.cli.train --config configs/toy_ctr_lsplm.yaml
