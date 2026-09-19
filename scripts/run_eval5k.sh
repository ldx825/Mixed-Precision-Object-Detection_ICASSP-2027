#!/bin/bash
# 5k re-evaluation of decisive Exp03v2 points (~75 min)
ROOT=/root/autodl-tmp/EviZO-VP/F-LBQ
cd "$ROOT"
.venv/bin/python src/diagnostics/eval5k_points.py > logs/eval5k.log 2>&1
echo EVAL5K_DONE >> logs/eval5k.log
