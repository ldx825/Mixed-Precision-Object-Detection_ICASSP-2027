#!/bin/bash
# Pair-extended chain: after v3 chain -> rerun exp54 with extended starts (uniform starts added),
# 6.0 first (biggest gap).
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
echo "[pair-ext] waiting for v3 chain..."
while pgrep -f run_v3_chain > /dev/null || pgrep -f exp55_v3_combined > /dev/null; do sleep 30; done
echo "[pair-ext] v3 done $(date)"
mkdir -p outputs/GateB/power_refined2/bak
for f in 6.0_pair.json 5.0_pair.json 5.5_pair.json; do
  [ -f outputs/GateB/power_refined2/$f ] && cp outputs/GateB/power_refined2/$f outputs/GateB/power_refined2/bak/$f
done
for TAG in 6.0 5.0 5.5; do
  nohup $PY src/gate_b/exp54_pair_swap.py $TAG > outputs/gateb_pairext_${TAG/./}_log.txt 2>&1 &
  PID=$!
  echo "[pair-ext] exp54 $TAG started PID $PID $(date)"
  wait $PID
  echo "[pair-ext] exp54 $TAG done rc=$? $(date)"
done
echo "PAIR_EXT_DONE $(date)" > outputs/pair_ext.done
echo "[pair-ext] ALL DONE"
