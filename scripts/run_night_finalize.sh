#!/bin/bash
# Night finalize chain:
#   wait exp40b -> refine+evaluate full-budget conditional (4.5, then 5.5)
#   -> wait main eval -> regenerate verdict
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[final] waiting for exp40b..."
while pgrep -f exp40b_conditional_full > /dev/null; do sleep 30; done
echo "[final] exp40b done $(date)"

if [ -f outputs/GateB/cond_full_45.json ]; then
  $PY src/gate_b/exp42_refine.py outputs/GateB/cond_full_45.json > outputs/gateb_refine45_log.txt 2>&1
  echo "[final] refine 4.5 done rc=$? $(date)"
else
  echo "[final] cond_full_45.json missing!"
fi

if [ -f outputs/GateB/cond_full_55.json ]; then
  $PY src/gate_b/exp42_refine.py outputs/GateB/cond_full_55.json > outputs/gateb_refine55_log.txt 2>&1
  echo "[final] refine 5.5 done rc=$? $(date)"
else
  echo "[final] cond_full_55.json missing!"
fi

echo "[final] waiting for main eval..."
while pgrep -f eval_allocations > /dev/null; do sleep 60; done
echo "[final] main eval done $(date)"

$PY src/gate_b/gate_b_verdict.py > outputs/gateb_verdict_log.txt 2>&1
echo "[final] verdict rc=$?"
echo "NIGHT_FINALIZE_DONE $(date)" > outputs/night_finalize.done
echo "[final] ALL DONE $(date)"
