#!/bin/bash
# Attack-4 chain: wait for exp61 -> exp62 (5.0/5.5 level-3 multi-start) -> exp63 (Swin-S deep push)
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[attack4] waiting for exp61..."
while pgrep -f exp61_swins_sweep > /dev/null; do sleep 60; done
echo "[attack4] exp61 done $(date)"
for S in exp62_3bit_multistart exp63_swins_refine; do
  echo "[attack4] $S start $(date)"
  $PY src/gate_b/$S.py > outputs/attack4_${S}.log 2>&1
  echo "[attack4] $S done rc=$? $(date)"
done
echo "ATTACK4_DONE $(date)" > outputs/attack4.done
