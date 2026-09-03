# M14 — the MNIST-temporal benchmark ladder
# Roadmap + exp-1 preregistration (draft for sign-off). Nothing run.

Replaces "another synthetic parity task" with real temporal
benchmarks that test increasingly meaningful notions of time:

    sMNIST  ->  N-MNIST  ->  Moving MNIST
    memory  ->  relationships  ->  prediction

## Why this ladder (reviewer)

- **sMNIST** (784 pixels fed one-per-tick, classify at the end):
  time is an artificial serialization of space. Tests whether
  temporal state can REPLACE spatial access — the literal "flip
  space into time" idea. Used here as a CONTROL / first sanity.
- **N-MNIST** (event-camera MNIST, asynchronous (x,y,t,polarity)
  over ~300 ms): relative event TIMING is genuinely part of the
  signal. The natural home for the relational-coordinate test.
- **Moving MNIST** (digits translating/bouncing, predict future
  frames): the object itself has dynamics; recurrence must
  represent VELOCITY, not just store the past.

## Shared frozen methodology (carried from M13.5 + step-2)

Same discipline for every experiment: equal cell count across
arms; identical input stream; convergence-BEFORE-performance
(dt vs dt/2 per-seed score stability, |Delta metric| < tol, before
reading which arm wins); null-adjusted skill; no tuning on test;
no feature engineering after seeing failures; topology seed = the
statistical unit; full feature/interface cost ledger. The
relational basis, wherever used, is frozen: `{cos(phi_i-phi_j),
sin(phi_i-phi_j) : i<j} ∪ {Kuramoto R}` (cos/sin, not wrapped
Δphi). Coordinate views compared at MATCHED feature dimension via
a PCA basis fit on calibration only.

---

# EXP-1 (sMNIST) — does matching temporal GEOMETRY matter more
# than raw state count? [detailed, offline-runnable]

## Question
Fixed cell budget, identical scalar pixel stream: does an
oscillatory substrate whose internal period is MATCHED to the
stimulus rate compute better than one that is not — and better
than non-oscillatory controls? A geometry/memory test, not yet a
coordinate-view test.

## Data (offline first pass, then scale)
Primary offline: **sMNIST-8** = sklearn `load_digits` (8x8, 1797
samples, 10 classes) flattened row-major to a length-64 scalar
stream, pixel intensities normalized to [0,1]. Deterministic, no
download. Split: fixed train/val/test (e.g. 1200/300/297).
Scale-up (later, needs data): full 28x28 sMNIST (784-length) and
psMNIST (one fixed frozen permutation).

## Arms (equal N cells, same stream, same readout)
`linear-trace bank | LIF | single oscillator (uncoupled) |
coupled oscillator`. The coupled arm additionally exposes the
frozen relational readout (its A vs R vs A+R views) so the
coordinate signal appears even here. N fixed for all.

## Presentation-speed sweep (the geometry probe)
Pixels held `T_pixel` internal ticks each, `T_pixel in {0.25, 0.5,
1, 2, 4}` relative to the oscillator's natural period. Frozen grid.
If an oscillatory arm has a preferred stimulus-rate:internal-period
ratio, accuracy peaks at a non-trivial T_pixel — direct evidence
that temporal geometry matters. Non-oscillatory controls are
expected to be flat/monotone in T_pixel.

## Readout + metric
10-way logistic (L2, C=1, lbfgs, tol=1e-4, class_weight balanced)
on the mean-pooled per-cell window features over the sequence
(and, for the coupled arm, on A / R / A+R views at matched dim).
Standardize by train stats. Metric: accuracy (chance 0.1) and
null-adjusted skill `S = max(0, (acc-0.1)/0.9)`. Frozen.

## Convergence before performance
At each arm's operating point, run integration substeps n vs 2n on
held-out seeds; require per-seed test-accuracy stability
`|Delta acc| < 0.03` BEFORE any arm-vs-arm reading. An arm whose
accuracy is not step-halving stable is reported as such and not
ranked.

## Pre-committed interpretation
- Oscillatory arms show a T_pixel resonance peak that controls lack
  -> matching temporal geometry buys computation. 
- Coupled >= single oscillator at the resonant T_pixel -> coupling
  (relational structure) adds beyond single-cell temporal tuning.
- All arms flat and equal at matched N -> geometry does not help
  here (honest null; sMNIST's artificial time may simply not
  reward it — which is why N-MNIST is the serious test).
- Within the coupled arm: R >= A at the resonant point would be an
  early positive for the relational-coordinate hypothesis, to be
  confirmed properly on N-MNIST.

## Frozen settings
N, seeds (calibration {0,1} / held-out test {10..15}), T_pixel
grid, readout, PCA-matched dimension, splits — all fixed before
running. Sanity-scale run (sMNIST-8) first; full sMNIST/psMNIST on
sign-off after the harness is shown well-posed.

---

# EXP-2 (N-MNIST) — the coordinate bridge [needs data; sketch]

Compress the 34x34 event field to N=16 or 32 spatial receptive
fields; each region drives one cell of the SAME physical network.
From identical trajectories build three views:
`R_state=[x_i]`, `R_events=[rates/counts]`,
`R_rel=[cos/sin(phi_i-phi_j)]`. Same input, same cells, same task;
only the representation changes. Linear classifier answers: does
relational timing expose the class more efficiently (per matched
dimension / per cost) than absolute state? This is the step-2
coordinate bridge (invariance -> decodability -> utility) on a
dataset where event timing is real. Pre-committed contrast vs
exp-1: relational advantage is EXPECTED to appear here and NOT on
sMNIST's artificial time. Requires N-MNIST data + an event loader
(`tonic` not installed) — a data-acquisition step to schedule.

# EXP-3 (Moving MNIST) — prediction [later; sketch]
Observe 5 frames, predict digit position 5 frames ahead
(`x(t)+xdot(t) -> x(t+Δ)`), and/or class + motion direction.
Recurrence must represent velocity. Tests temporal EVOLUTION, the
top of the ladder.

## Status
Draft for reviewer sign-off. On sign-off, exp-1 (sMNIST-8) runs
first — offline, decisive on well-posedness and the geometry
question — before any dataset download for exp-2.
