#!/bin/bash
# GateB chain: wait signals -> Exp40 conditional -> Exp35 seed validation -> static allocations -> 5k eval
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[gateb-chain] waiting for measure_signals ..."
while pgrep -f measure_signals.py > /dev/null; do sleep 10; done
echo "[gateb-chain] signals done at $(date)"

$PY src/gate_b/exp40_conditional.py > outputs/gateb_cond_log.txt 2>&1
echo "[gateb-chain] exp40 (conditional greedy) done rc=$? at $(date)"

$PY src/diagnostics/exp35_seed_pair_validation.py > outputs/exp35_log.txt 2>&1
echo "[gateb-chain] exp35 (seed validation) done rc=$? at $(date)"

$PY src/gate_b/allocate_static.py > outputs/gateb_static_log.txt 2>&1
echo "[gateb-chain] allocate_static done rc=$? at $(date)"

nohup $PY src/gate_b/eval_allocations.py > outputs/gateb_eval_log.txt 2>&1 &
echo "[gateb-chain] eval_allocations launched PID $! (5k evals, 5.0-budget first)"
echo "[gateb-chain] ALL LAUNCHED at $(date)"
