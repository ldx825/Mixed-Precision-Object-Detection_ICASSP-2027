#!/bin/bash
# Pair-swap chain: wait for shgreedy chain -> exp54 for 4.5 -> 5.5 -> 5.0
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[pair] waiting for shgreedy chain..."
while pgrep -f run_shgreedy_chain > /dev/null || pgrep -f exp53_sh_greedy > /dev/null; do sleep 30; done
echo "[pair] shgreedy chain done $(date)"
for TAG in 4.5 5.5 5.0 6.0; do
  nohup $PY src/gate_b/exp54_pair_swap.py $TAG > outputs/gateb_pair${TAG/./}_log.txt 2>&1 &
  PID=$!
  echo "[pair] exp54 $TAG started PID $PID $(date)"
  wait $PID
  echo "[pair] exp54 $TAG done rc=$? $(date)"
done
echo "PAIR_CHAIN_DONE $(date)" > outputs/pair_chain.done
echo "[pair] ALL DONE"
