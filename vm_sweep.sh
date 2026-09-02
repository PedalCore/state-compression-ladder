#!/bin/bash
set -e
cd "$(dirname "$0")"
[ -f results/hh_data_full.npz ] || python3 hh_teacher.py
for k in 1 2 4; do
  for eh in 0.3 0.6 1.0; do
    for s in 0 1; do
      python3 hh_ssm.py --k $k --eps-hi $eh --seed $s --dev cuda
    done
  done
done
echo "=== VM SWEEP DONE OK ==="
