#!/bin/bash
# Split-half validated refinement chain: exp52 for 4.5 -> 5.5 -> 5.0
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
for TAG in 4.5 5.5 5.0; do
  nohup $PY src/gate_b/exp52_split_half.py $TAG > outputs/gateb_xl${TAG/./}_log.txt 2>&1 &
  PID=$!
  echo "[xl] exp52 $TAG started PID $PID $(date)"
  wait $PID
  echo "[xl] exp52 $TAG done rc=$? $(date)"
done
echo "XL_CHAIN_DONE $(date)" > outputs/xl_chain.done
echo "[xl] ALL DONE"
