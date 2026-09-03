# M14 Exp-2 (N-MNIST) — preprocessing FREEZE
# Frozen BEFORE any classification result is seen. The coordinate
# bridge: does information migrate from cell STATES into
# RELATIONSHIPS between cells when event timing is real?

## Acquisition is a DEPENDENCY, not part of the experiment

Tonic is used ONCE to download/parse N-MNIST, then each example is
written to a NEUTRAL cached format the experiment consumes:
per sample `(x, y, t, p, label)` raw arrays (int), plus a manifest
with dataset name, tonic version, per-split sample counts, and a
sha256 over the concatenated raw arrays. `m14_nmnist.py` reads ONLY
the cache — never tonic — so a future tonic transform/default
change cannot silently alter M14. Sensor is 34x34x2 (ON/OFF);
events returned in native `(x,y,t,p)` order.

## What is NOT done (would change the question)
- **No ToFrame / temporal binning** for the primary experiment —
  framing quantizes away the event timing we are testing. The only
  discretization permitted is whatever the dynamical integration
  itself requires (dt), and that is convergence-tested.
- **No saccade stabilization** (`stabilize=True` OFF). The camera
  motion IS the source of N-MNIST's temporal structure; removing it
  removes the phenomenon.
- **No polarity summing before the cells.** ON/OFF timing is
  genuine information.

## Frozen preprocessing pipeline
1. **Timestamp scaling:** subtract per-recording origin `t -= t.min()`
   (translation only); scale to the frozen substrate timescale by a
   single global constant `t_scale` (ms -> T0 units) chosen ONCE
   from the median recording duration, frozen. No per-sample
   rescaling.
2. **Receptive fields:** partition the 34x34 grid into `N` fixed
   non-overlapping tiles (N in {16, 32}; 16 = a 4x4 tiling of
   ~8x8 blocks). Tiling fixed before results; each event routes to
   its tile by (x,y). Frozen.
3. **Polarity-preserving drive:** each cell i receives
   `I_i(t) = g_plus * E_i^+(t) - g_minus * E_i^-(t)`, where
   `E_i^{+/-}(t)` are the ON/OFF event streams into tile i,
   delivered as instantaneous kicks at native event times (or a
   fixed narrow exponential kernel), affine and frozen. ON and OFF
   are NOT summed before the cell.
4. **Substrate:** the SAME lightweight phase system as Exp-1
   (uncoupled g_rec=0 and coupled g_rec>0), driven by `I_i(t)`.
   HH/M13 stay OUT (step 3). Same g_rec=0==uncoupled invariant,
   unit-tested first.

## Three coordinate VIEWS (identical trajectories; only the view differs)
- `R_state  = [cos(theta_i)]`            (absolute), dim N
- `R_events = [event rate / count per tile]` (absolute, event-domain)
- `R_rel    = [cos(theta_i-theta_j), sin(theta_i-theta_j)]` over
  exactly N/2 preselected pairs (fixed from the reservoir seed) so
  `dim R_rel = N`. A+R via the frozen PCA->N matched-dimension rule.
Kuramoto R_K diagnostic only, not in the primary classifier.

## Order of operations (reviewer-signed)
1. Freeze everything above (this doc).
2. dt-CONVERGENCE / representation gate FIRST: at the frozen
   receptive-field + polarity mapping, run integration dt vs dt/2
   on held-out samples; require per-sample-averaged classification
   score stability (|Delta acc| < 0.03) AND the g_rec=0 invariant,
   assessed only where acc is materially above chance (the SeqDigits
   metric refinement). Only if it passes:
3. CLASSIFY: 10-way logistic (F5 rules), matched-dimension views,
   report R_state vs R_events vs R_rel vs A+R, topology seed = unit,
   every seed reported.

## Pre-committed prediction (the weight of M14)
- sMNIST/SeqDigits: coupling may help; relational coordinates need
  NOT (artificial time). [SeqDigits-64: coupling helped, confirmed.]
- **N-MNIST: if M14 is right, `R_rel` becomes SPECIFICALLY useful —
  R_rel >= R_state at matched dimension — i.e. the information has
  migrated from cell states into relationships.** Falsified if
  R_rel <= R_state after matching, or if it fails the convergence
  gate.

## Status
Preprocessing frozen. Acquisition (tonic -> neutral cache) and the
convergence gate run before any classification number is read.

---
# EXP-2 RESULT (2026-09-03) — frozen relational-final-state bridge
# FAILS the convergence gate on N-MNIST. Honest negative. Fork.

Neutral cache built (2000 train / 1000 test, balanced, checksummed;
sensor 34x34x2, t in us ~305ms, polarity {0,1} preserved, no
framing, no stabilize). g_rec=0 invariant exact (0). N=16 tiles.

CONVERGENCE GATE (|Δacc|, nbin vs 2*nbin), coupled, BEFORE classify:
                     nbin=60          nbin=120         verdict
  R_state (cosθ)     0.28/0.35 Δ.067  0.35/0.45 Δ.105  UNSTABLE
  R_events (rate)    0.71/0.71 Δ.000  0.71/0.71 Δ.000  stable
  R_rel  (cos/sinΔθ) 0.26/0.32 Δ.061  0.32/0.39 Δ.076  UNSTABLE
Same with an impulsive drive AND with a frozen exponential kernel
(TAU_K=0.5): the kernel did NOT rescue it. Accuracy CLIMBS
monotonically with finer dt (0.28->0.35->0.45) — the phase-final-
state readout is not converging; halving dt keeps changing it.

DIAGNOSIS. The only stable view is R_events — a STATIC spatial
event-rate histogram (no dynamics), trivially dt-independent, and
it dominates at 0.711. The phase-coordinate views (absolute
R_state and relational R_rel) read the FINAL phase, which
accumulates dt-dependent drift over the ~5-period recording; the
cells are event-driven and do NOT phase-lock, so the relative
phases drift differently and R_rel does not cancel the drift. This
is the M13.5 discretization-fragility lesson, reproduced on real
event data: a final-state phase readout of a non-locking driven
network is not step-halving stable.

RECONCILIATION WITH v0 (important). v0's STABLE relational quantity
was the TIME-AVERAGED synchronization of a CONSTANT-driven, phase-
LOCKING network (Kuramoto R corr 0.997). Exp-2's UNSTABLE quantity
is the FINAL-STATE phase of an EVENT-driven, NON-locking network.
So v0 and Exp-2 are consistent: relational structure is robust when
it is time-averaged over a locking regime, fragile when read as a
single accumulated final phase of a non-locking one.

DECISION (pre-committed tree): R_rel FAILS the convergence gate ->
the frozen relational-final-state bridge is NOT well-posed on
N-MNIST -> honest negative. No classification ranking of R_rel vs
R_state is read (both non-converged). R_events (spatial rate 0.71)
is the stable baseline and the number to beat.

FORK (do NOT self-rescue mid-run; a NEW preregistration, like
M13.5 P2->P2b): Exp-2b = TIME-AVERAGED relational readout (the
v0-stable observable: window-/recording-averaged cos/sin Δθ and
Kuramoto R over the trajectory, not the final snapshot), possibly
with a stronger-coupling locking regime, re-gated for convergence
BEFORE classification. Question unchanged: at matched dimension,
does a time-averaged relational view expose the class better than
absolute state — i.e. does information live in the relationships?
Reviewer to choose Exp-2b vs close the relational bridge as
"well-posed only for time-averaged locking observables."
