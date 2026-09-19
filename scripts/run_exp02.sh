#!/bin/bash
# Exp02 — quantization sanity tests A-D (plan §16)
set -e
ROOT=/root/autodl-tmp/EviZO-VP/F-LBQ
cd "$ROOT"
mkdir -p logs
.venv/bin/python src/quant/sanity.py "$@" 2>&1 | tee logs/exp02.log
