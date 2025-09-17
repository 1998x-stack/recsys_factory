#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
cd data
if [ ! -d "ml-100k" ]; then
  echo "[download] MovieLens-100K..."
  curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o ml-100k.zip
  unzip -q ml-100k.zip
  rm -f ml-100k.zip
  echo "[ok] data/ml-100k/u.data ready"
else
  echo "[skip] data/ml-100k exists"
fi
