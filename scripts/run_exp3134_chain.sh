#!/bin/bash
# Chain: wait for Exp30b -> run offline analyses (Exp31, Exp33) -> launch Exp34 (GPU seed curve).
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
echo "[chain] waiting for exp30b..."
while pgrep -f exp30b_full_subsets.py > /dev/null; do sleep 10; done
echo "[chain] exp30b finished at $(date)"

PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
$PY src/diagnostics/exp31_context_rank_reversal.py > outputs/exp31_run.log 2>&1
echo "[chain] exp31 done ($?)"
$PY src/diagnostics/exp33_decision_k4.py > outputs/exp33_run.log 2>&1
echo "[chain] exp33 done ($?)"

nohup $PY src/diagnostics/exp34_seed_curve.py > outputs/exp34_log.txt 2>&1 &
echo "[chain] exp34 launched PID $!"
echo "[chain] ALL DONE at $(date)"
