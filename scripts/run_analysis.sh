#!/bin/bash
# Post-sweep analysis pipeline: eval probe results -> correlation report + figure
ROOT=/root/autodl-tmp/EviZO-VP/F-LBQ
cd "$ROOT"
.venv/bin/python src/diagnostics/eval_sweep.py --sweep-dir outputs/Exp03 --out-prefix Exp03 2>&1 | tee logs/eval_sweep.log
.venv/bin/python src/diagnostics/exp04_correlations.py 2>&1 | tee logs/exp04.log
echo ANALYSIS_DONE > logs/analysis.done
