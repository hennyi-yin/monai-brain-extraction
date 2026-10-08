#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.make_split
python -m src.qc
if [[ -f checkpoints/last.pt ]]; then
  python train.py --resume
else
  python train.py
fi
python -m src.run_bet
python evaluate.py
