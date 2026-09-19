#!/bin/bash
# Power-refine chain: finish current 4.5 -> 5.5 (from refined bits) -> 5.0 (from oracle bits)
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[power-chain] waiting for current exp50..."
while pgrep -f exp50_power_refine > /dev/null; do sleep 30; done
echo "[power-chain] 4.5 done $(date)"

$PY src/gate_b/exp50_power_refine.py outputs/GateB/refined_cond_full_55_results.json 5.5 > outputs/gateb_power55_log.txt 2>&1
echo "[power-chain] 5.5 done rc=$? $(date)"

$PY src/gate_b/exp50_power_refine.py outputs/GateB/allocations_oracle.json 5.0 > outputs/gateb_power50_log.txt 2>&1
echo "[power-chain] 5.0 done rc=$? $(date)"

echo "POWER_CHAIN_DONE $(date)" > outputs/power_chain.done
echo "[power-chain] ALL DONE"
