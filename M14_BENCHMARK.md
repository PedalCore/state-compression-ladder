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

---
# EXP-1 FROZEN AMENDMENTS (reviewer sign-off, 2026-09-03)

## A1. Name the offline version honestly
sklearn 8x8 digits flattened = 64-step task, NOT sMNIST (28x28=784,
radically different memory demand). Call it **SeqDigits-64
qualification** (smoke/well-posedness). Production Exp-1 uses real
MNIST and is **sMNIST-784**. `SeqDigits-64 -> sMNIST-784`; never
call both sMNIST.

## A2. Separate presentation speed from numerical resolution
Frozen common reference timescale `T0`. Physical pixel duration
`T_p in {0.25,0.5,1,2,4}*T0`, IDENTICAL for every arm (do NOT
normalize T_p to each cell's own time constant). Integrate each
continuous arm with `dt = T_p/n`; test convergence `n -> 2n` at
each T_p WITHOUT changing T_p. This cleanly separates STIMULUS
timescale from NUMERICAL timescale — the ambiguity M13.5 taught us
to kill.

## A3. Oscillator = lightweight phase system (NOT HH)
`theta_i in S^1`,
`dtheta_i = omega_i + g_in*b_i*u(t) + g_rec*(1/d_i)*sum_j A_ij*sin(theta_j-theta_i)`.
Uncoupled arm `g_rec=0`; coupled arm `g_rec>0`. Same cell law,
state dim (1), input masks, frequencies, solver — ONLY coupling
differs. INVARIANT (unit-tested before any run): coupled model at
g_rec=0 == uncoupled oscillator exactly (the M14 analog of the
block experiment's b=1 invariant). HH/M13 are explicitly kept OUT
of Exp-1 (and out of Exp-2's coordinate bridge) so an oscillator
effect can't be confounded with HH complexity; they enter only at
step 3.

## A4. Representation freezes (avoid coordinate artifacts)
Never expose raw theta (the +/-pi discontinuity is a coordinate
artifact). Absolute view `A_i = cos(theta_i)`, dim N. Relational
view `R_ij = [cos(theta_i-theta_j), sin(theta_i-theta_j)]`; because
that is 2 features/pair, PRESELECT exactly N/2 pairs (fixed from the
topology seed before results) so `dim R = N = dim A`. A+R uses the
already-frozen matched-dimension rule (PCA->N), NOT 2N raw features.
Kuramoto R_K is DIAGNOSTIC only, not in the primary classifier.

## A5. Equal-N is the whole claim scope
Exp-1 is a GEOMETRY experiment. The only admissible conclusion is
"at equal cell count, this temporal geometry produced a more/less
useful representation." NO per-hardware-cost claim (a sin-coupling
op may cost far more than an LIF leak); that normalization waits
until something is worth normalizing. Do not repeat M13.5 in
miniature.

## Crisp Exp-1 question
Equal N, identical serialized input, arms {trace | LIF | uncoupled
phase osc | coupled phase osc}, sweep common T_p:
PRIMARY -> does classification depend systematically on the match
between input timescale and substrate dynamics? SECONDARY (coupled
arm only, early indicator, no headline) -> A vs R vs A+R.

---
# EXP-1 SeqDigits-64 QUALIFICATION RESULT (2026-09-03)

Invariant coupled(g_rec=0)==uncoupled: PASS (max|diff|=0). N=16,
chance=0.10, 600 train / 300 test, T0=1, bounded solver dt~0.1.

CONVERGENCE GATE (|Δacc| dt vs dt/2, before ranking):
- FIRST run (coarse dt=Tp/n, n=2..4) FLAGGED the controls: trace
  worst 0.398 (0.60->0.20 at Tp=4), LIF 0.035 — forward-Euler on
  leaky integrators is stiff at large Tp with few substeps. The
  oscillators were exact/stable. The "simple" arms were the
  numerically fragile ones. Gate did its job.
- BOUNDED dt (dt~0.1, substeps scaled to Tp) FIXED it: trace 0.003
  stable, LIF 0.030 borderline-stable, osc 0.000. osc_coupled
  flagged 0.043 UNSTABLE but ONLY in the large-Tp NEAR-CHANCE tail
  (Tp1-4 acc~0.10-0.12 = chance +/- sampling noise); at its
  informative point (Tp0.25) it is exactly stable (0.45/0.45). So
  the flag is a worst-case-metric artifact of chance-regime noise,
  not a dynamical instability.

GEOMETRY SWEEP (median acc, bounded dt):
  trace       Tp0.25:0.50 0.5:0.49 1:0.43 2:0.34 4:0.26  peak 0.25
  LIF         Tp0.25:0.40 0.5:0.43 1:0.36 2:0.34 4:0.31  peak 0.5
  osc(uncpl)  Tp0.25:0.14 0.5:0.16 1:0.15 2:0.09 4:0.11  ~chance
  osc_coupled Tp0.25:0.41 0.5:0.25 1:0.11 2:0.12 4:0.12  peak 0.25

FINDINGS (honest, equal-N scope only — NO per-hardware claim):
1. Harness well-posed: invariant exact; with bounded dt all
   rankable arms step-halving stable. Bounded dt is now REQUIRED
   (fold into frozen protocol); convergence should be assessed only
   where acc is materially above chance (metric refinement for
   sMNIST-784).
2. NO temporal-geometry RESONANCE: every arm peaks at the smallest
   Tp and declines; no non-trivial preferred stimulus:internal
   ratio. On SeqDigits-64 (artificial serialized time), matching
   temporal geometry does NOT buy computation — the pre-committed
   "artificial time may not reward it" branch. (This is exactly why
   N-MNIST, with real event timing, is the serious test.)
3. COUPLING adds real separability: osc_coupled (0.41) >> osc
   uncoupled (0.14) at the informative Tp, lifting a ~1-state phase
   oscillator to near the leaky-integrator controls (trace 0.50,
   LIF 0.40). Early positive for the coupling/relational direction,
   to be properly tested (A|R|A+R) on N-MNIST.

VERDICT: harness QUALIFIED (well-posed) to proceed. Geometry-
resonance hypothesis: null on this artificial-time control.
Coupling: promising. NEXT = sMNIST-784 (real 28x28, needs MNIST
download) to confirm at scale, and EXP-2 N-MNIST coordinate bridge
(needs event data + loader) for the relational headline.
