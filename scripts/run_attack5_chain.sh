#!/bin/bash
# Attack-5: wait for attack4 -> exp64 (baseline fairness with {3..8} level set)
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[attack5] waiting for attack4..."
while [ ! -f outputs/attack4.done ]; do sleep 60; done
echo "[attack5] attack4 done $(date)"
$PY src/gate_b/exp64_baseline_3bit_fairness.py > outputs/attack5_exp64.log 2>&1
echo "[attack5] exp64 done rc=$? $(date)"
echo "ATTACK5_DONE $(date)" > outputs/attack5.done
