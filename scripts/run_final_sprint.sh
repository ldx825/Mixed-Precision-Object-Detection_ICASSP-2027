#!/bin/bash
# Final sprint chain: exp51 for 4.5 -> 5.5 -> 5.0
cd /root/autodl-tmp/EviZO-VP/F-LBQ || exit 1
PY=/root/autodl-tmp/EviZO-VP/F-LBQ/.venv/bin/python
for TAG in 4.5 5.5 5.0; do
  nohup $PY src/gate_b/exp51_final_sprint.py $TAG > outputs/gateb_sprint${TAG/./}_log.txt 2>&1 &
  PID=$!
  echo "[sprint] exp51 $TAG started PID $PID $(date)"
  wait $PID
  echo "[sprint] exp51 $TAG done rc=$? $(date)"
done
echo "FINAL_SPRINT_DONE $(date)" > outputs/final_sprint.done
echo "[sprint] ALL DONE"
