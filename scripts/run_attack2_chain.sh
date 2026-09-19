#!/bin/bash
# Attack-2 chain: exp56 (full-budget greedy @5.0/6.0) -> exp56c (budget completion) -> exp57 (4.0 3-bit)
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
for S in exp56_full_greedy_5060 exp56c_budget_completion exp57_3bit_ext_40; do
  echo "[attack2] $S start $(date)"
  $PY src/gate_b/$S.py > outputs/attack2_${S}.log 2>&1
  echo "[attack2] $S done rc=$? $(date)"
done
echo "ATTACK2_DONE $(date)" > outputs/attack2.done
