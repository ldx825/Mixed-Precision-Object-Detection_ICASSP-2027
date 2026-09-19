#!/bin/bash
# Attack-3 chain: exp59 (level-3 extension @4.5/5.0/5.5) -> exp60 (Swin-S second backbone)
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
for S in exp59_3bit_ext_455055 exp60_swins_baseline; do
  if [ ! -f src/gate_b/$S.py ]; then echo "[attack3] src/gate_b/$S.py missing; skipping"; continue; fi
  echo "[attack3] $S start $(date)"
  $PY src/gate_b/$S.py > outputs/attack3_${S}.log 2>&1
  echo "[attack3] $S done rc=$? $(date)"
done
echo "ATTACK3_DONE $(date)" > outputs/attack3.done
