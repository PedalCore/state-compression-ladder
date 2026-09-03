#!/bin/bash
set -e; cd "$(dirname "$0")/.."
python3 hh_teacher.py --seed 0            # train/val/dev corpus
python3 hh_teacher.py --seed 777 --final  # LOCKED held-out corpus
