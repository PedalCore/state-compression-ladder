#!/bin/bash
set -e; cd "$(dirname "$0")/.."
# stage-0 fields (frozen bases) are produced by hh_comp seed 0/1;
# then the trust-hybrid correctors (A4/A1-fixed):
for s in 0 1; do python3 hh_comp.py --seed $s; done
for kc in 8 1; do for s in 0 1; do
  python3 hh_hybrid.py --seed $s --kc $kc --lam 0.1 --trust
done; done
