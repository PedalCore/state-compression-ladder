# Reproducibility path — M13 state-compression ladder

The clean path through the repository for the paper's headline
numbers. (The full exploratory history is in NOTEBOOK.md; this
directory is the minimal reproduce-the-claims route.)

    bash repro/make_teacher.sh     # HH teacher + LOCKED corpus
    bash repro/train_headline.sh   # field + trust-hybrid (kc 8,1)
    bash repro/locked_eval.sh      # single-shot held-out eval

`numbers.json` maps every reported paper number to its source
checkpoint/result JSON. Seeds and configs are fixed in each
script. `env.txt` pins the environment.

Headline held-out (locked corpus, seed 777, single-shot):
  trust-hybrid kc=8 : F1 0.879 / 0.888  (seeds 0/1)
  trust-hybrid kc=1 : F1 0.861 / 0.836  (32 params, 1 scalar)
  observable parallel baseline : ~0.264
All with correct anodal-break rebound; f-I 0.3-0.6 Hz
(kc=1 seed 1: 9.9 Hz — reported, not averaged away).
