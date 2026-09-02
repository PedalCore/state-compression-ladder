#!/bin/bash
set -e
cd "$(dirname "$0")"
[ -f results/hh_data_full.npz ] || python3 hh_teacher.py
for kind in ssm gru; do
  for k in 1 2; do
    for s in 0 1; do
      python3 hh_joint.py --kind $kind --k $k --seed $s --dev cuda
    done
  done
done
echo "=== VM JOINT DONE OK ==="
