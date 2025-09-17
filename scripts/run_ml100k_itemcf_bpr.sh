#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS=ignore
echo "[run] ItemCF→BPR→MMR on ML-100K"
python -m recsys_factory.cli.train --config configs/ml100k_itemcf_bpr.yaml
