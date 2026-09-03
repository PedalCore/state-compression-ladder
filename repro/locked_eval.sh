#!/bin/bash
set -e; cd "$(dirname "$0")/.."
python3 repro/locked_eval.py
