#!/bin/bash
# Exp03 — single-group perturbation sweep on probe512 (plan §19-20)
# Usage: nohup bash scripts/run_exp03_sweep.sh > logs/exp03_nohup.log 2>&1 &
ROOT=/root/autodl-tmp/EviZO-VP/F-LBQ
cd "$ROOT"
mkdir -p logs
.venv/bin/python src/diagnostics/sweep_groupbits.py \
  --bits 6,4 --probe 512 --calib 64 --workers 6 --resume "$@" 2>&1 | tee -a logs/exp03_sweep.log
