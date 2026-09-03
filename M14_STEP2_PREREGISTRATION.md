# M14 step-2 — do relational COORDINATES turn stable collective
# dynamics into useful, converged computation?
# Preregistration. Draft for reviewer sign-off. Nothing run.

## The one question (narrow — NOT yet M13 vs LIF)

> Do relational coordinates turn stable collective dynamics into
> **useful, converged computation**?

This is a COORDINATE-SYSTEM experiment on a single substrate, not a
neuron-model comparison. "Does richer local dynamics build a better
relational basis?" is step 3 and is explicitly out of scope here.

## What v0 did and did NOT establish

v0 showed relational observables (Kuramoto R, pairwise phase) are
far LESS sensitive to numerical timing drift than absolute spike
times (diff 5e-4, corr 0.997 vs absolute 0.03->0.10 ms). It did NOT
show those observables carry USEFUL COMPUTATIONAL INFORMATION. Step
2 is exactly the missing bridge:
`invariance -> decodability -> task utility`.

## Design: one dynamical system, multiple readout VIEWS

ONE coupled HH network (the v0 substrate). Same trajectories, same
topology W, same symbol-held input, same seeds. Input is symbol-
held `u_k` modulating the drive so it perturbs relative phases:
`I_i(t) = I0 + spread*b_i + g_in*b_i*u_k + gc*(W@z_i)`. From the
SAME trajectories we build three coordinate VIEWS and change ONLY
the coordinate system:
- **A (absolute):** per-cell window-mean observable `r^A_i[k]`
  (identical to the M13.5/P2 absolute feature). Dimension N.
- **R (relational) — PRIMARY:** window-mean of the frozen
  relational basis (below). 
- **A+R:** concatenation.
Because all three views are read from identical trajectories, the
substrate operating point is a property of the SUBSTRATE, not the
arm — selected once (below), frozen, shared by all views.

## Relational basis — frozen in advance

Per-cell phase `phi_i(t)` by linear interpolation between
successive spike times (v0 method). Primary pairwise features per
symbol window:
`cos(phi_i - phi_j)` and `sin(phi_i - phi_j)` for all i<j,
window-averaged per symbol. (cos/sin, NOT raw wrapped Δphi — avoids
the ±π discontinuity.) One additional GLOBAL feature: the window-
mean Kuramoto R. NO other synchronization statistics may be added
after seeing results — the basis is exactly {cos Δphi_ij, sin
Δphi_ij : i<j} ∪ {R}. Frozen.

## Feature-count control (so "relational wins" != "more coordinates")

Pairwise features scale O(N^2): for N cells, R exposes N(N-1)+1
features vs A's N. Two reportings, both preregistered:
1. **PRIMARY — matched dimension.** Project EVERY view to a common
   K = N components via PCA fit ON CALIBRATION DATA ONLY (basis
   frozen, applied unchanged to val/test). So A, R, A+R each expose
   exactly K=N features; the only difference is which coordinate
   system was projected. This is the fair head-to-head.
2. **Secondary — full dimension + ledger.** Report raw full-dim
   scores WITH the complete feature/interface cost ledger
   (feature counts, phase-extraction ops, pair count) so any raw-
   dim advantage is visible as the cost it is.

## Tasks (continuity with the M13.5 absolute-coordinate probe)

Delayed recall `y_k = u_{k-d}` (ridge, S_recall=max(0,R^2)) and
2-bit delayed parity `y_k = sign(u_{k-d} u_{k-2d})` (logistic,
S_parity=max(0,2(acc-0.5))). Delays d in {2, 8} for the gate;
full ladder {1,2,4,8,16} only if the gate passes. Input
`u_k ~ U{-1,+1}`. Readouts EXACTLY the M13.5 F5 rules (ridge
lambda=1e-2; balanced logistic C=1, lbfgs, tol=1e-4; train-set
standardization). Null = frozen circular-shift baseline.

## Substrate operating point — skill-NEUTRAL selection, frozen

Selected ONCE on calibration by a rule that never sees task skill:
require (a) all cells firing (>= 5 spikes over the window), (b)
mean Kuramoto R in the partial-sync band [0.2, 0.6] (structure
present, not fully locked/incoherent), (c) dt-halving stable
network per v0 criterion. Among qualifying `(I0, g_in, gc)` grid
points pick the one with median R closest to 0.4; freeze. Same
point for all three views (they share trajectories).

## Convergence BEFORE performance (the real test)

FIRST, at the frozen point on HELD-OUT topologies, run substeps
dt vs dt/2 and require PER-SEED task-score stability for the
PRIMARY relational arm: `|Delta R^2| < 0.03` (recall) and
`|Delta acc| < 0.03` (parity) at every seed and delay (F3+F6, not
mean-only). The interesting question is whether v0's strong
observable-level invariance SURVIVES through a trained linear
decoder. Only if it converges do we read performance.

## Arms (relational-only PRIMARY, so the hypothesis stays falsifiable)

`A (absolute) | R (relational, PRIMARY) | A+R`. Interpretation,
pre-committed:
- `R converges AND reaches useful skill (recall R^2>0.2 OR parity
  >0.55)` -> relational coordinates turn stable dynamics into
  useful converged computation. HYPOTHESIS SUPPORTED.
- `R converges but ~null skill` -> invariant but NOT informative;
  the bridge fails at decodability. Honest negative.
- `R fails per-seed convergence` -> observable-level invariance
  does not survive decoding. Honest negative.
Then, among converged useful arms:
- `R > A` -> relative timing is a better coordinate than absolute.
- `A+R > R` -> relational structure helps but is not everything.
- `R ~= A+R` -> the useful computation lives essentially in the
  relationships (the striking outcome).
- `R < A` -> relational compactness costs task information here.

## Frozen settings + discipline

N, Th=1ms, calibration topology seeds {0,1}, held-out test seeds
{10..15}, splits + lengths as P2b. NO tuning on test; phase
extraction, pair selection, PCA basis, regularization and null
baselines all frozen before running. No feature engineering after
seeing any failure. Statistical unit = topology seed; report every
seed.

## Status
Draft for reviewer sign-off. Step 3 (richer local dynamics -> better
relational basis, at matched network size + relational interface)
becomes scientifically clean ONLY if step 2's bridge holds.
