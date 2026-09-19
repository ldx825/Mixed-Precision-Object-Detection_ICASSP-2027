#!/bin/bash
# SH-greedy chain: wait for exp52 (xl) chain -> run exp53 for 4.5 -> 5.5 -> 5.0
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[shg] waiting for xl chain..."
while pgrep -f run_xl_chain > /dev/null || pgrep -f exp52_split_half > /dev/null; do sleep 30; done
echo "[shg] xl chain done $(date)"
for TAG in 4.5 5.5 5.0 6.0; do
  nohup $PY src/gate_b/exp53_sh_greedy.py $TAG > outputs/gateb_shg${TAG/./}_log.txt 2>&1 &
  PID=$!
  echo "[shg] exp53 $TAG started PID $PID $(date)"
  wait $PID
  echo "[shg] exp53 $TAG done rc=$? $(date)"
done
echo "SHGREEDY_CHAIN_DONE $(date)" > outputs/shgreedy_chain.done
echo "[shg] ALL DONE"
