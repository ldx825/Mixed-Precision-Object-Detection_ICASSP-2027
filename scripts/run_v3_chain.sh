#!/bin/bash
# V3 chain: wait for pair chain -> exp55 (epsilon-tolerant combined) for 4.5/5.5/5.0/6.0
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[v3] waiting for pair chain..."
while pgrep -f run_pair_chain > /dev/null || pgrep -f exp54_pair_swap > /dev/null; do sleep 30; done
echo "[v3] pair chain done $(date)"
for TAG in 4.5 5.5 5.0 6.0; do
  nohup $PY src/gate_b/exp55_v3_combined.py $TAG > outputs/gateb_v3_${TAG/./}_log.txt 2>&1 &
  PID=$!
  echo "[v3] exp55 $TAG started PID $PID $(date)"
  wait $PID
  echo "[v3] exp55 $TAG done rc=$? $(date)"
done
echo "V3_CHAIN_DONE $(date)" > outputs/v3_chain.done
echo "[v3] ALL DONE"
