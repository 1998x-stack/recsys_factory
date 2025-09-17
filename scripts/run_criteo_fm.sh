#!/usr/bin/env bash
set -euo pipefail
echo "[run] CTR: FM on Criteo CSV (ensure data/criteo_small.csv exists)"
python -m recsys_factory.cli.train --config configs/criteo_fm.yaml
