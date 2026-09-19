#!/bin/bash
# Final collect: wait for pair-ext chain -> run final collector -> FINAL_TABLE.md
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[collect] waiting for pair-ext chain..."
while pgrep -f run_pair_ext_chain > /dev/null || pgrep -f exp54_pair_swap > /dev/null; do sleep 60; done
sleep 15
$PY src/gate_b/collect_final_results.py > outputs/collect_final_log.txt 2>&1
echo "[collect] done rc=$?"
echo "COLLECT_DONE $(date)" > outputs/collect.done
