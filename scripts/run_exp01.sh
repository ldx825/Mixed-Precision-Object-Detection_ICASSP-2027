#!/bin/bash
# Exp01 — FP baseline on full COCO val2017 (Gate: deviation >1.0 AP => STOP)
set -e
ROOT=/root/autodl-tmp/EviZO-VP/F-LBQ
cd "$ROOT"
mkdir -p logs
.venv/bin/python src/baseline/exp01_repro_fp.py --workers 6 --save-every 500 "$@" 2>&1 | tee logs/exp01.log
