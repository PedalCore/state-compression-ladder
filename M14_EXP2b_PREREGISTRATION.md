# M14 Exp-2b — TIME-AVERAGED relational readout of a LOCKING net.
# Preregistration. FROZEN before any classification. Last N-MNIST
# relational-bridge attempt (no rescue after, like M13.5 P2b).

## Why 2b (motivated, not a rescue)

Exp-2's frozen FINAL-STATE phase readout failed the convergence
gate: accumulated phase of an event-driven, NON-locking network is
dt-fragile. But v0's STABLE relational quantity was the TIME-
AVERAGED synchronization of a CONSTANT-driven, phase-LOCKING net
(Kuramoto R corr 0.997). Exp-2b tests exactly the v0-stable
observable on N-MNIST: **does a TIME-AVERAGED relational readout of
a LOCKING network turn real event timing into useful, CONVERGED
computation?** Same neutral cache, same discipline.

## Held fixed from Exp-2 (frozen)
Neutral N-MNIST cache only (never tonic); N=16 receptive-field
tiling; polarity-preserving EXPONENTIAL-kernel drive (TAU_K=0.5,
the smooth continuous-time drive); same lightweight phase substrate
with the g_rec=0==uncoupled invariant (unit-tested first). No
framing, no saccade-stabilization, polarity preserved.

## CHANGE 1 — locking operating point (skill-NEUTRAL, frozen)
Exp-2 was weakly coupled and did not lock. Select the coupling /
drive regime ONCE on calibration by a rule that never sees task
accuracy: over a predefined grid `g_rec in {2,4,8}` x
`gain_scale in {0.5,1,2}` (scales G_ON,G_OFF), compute the
recording-time-mean Kuramoto R per sample; require median R in the
partial-locking band `[0.3, 0.7]`; among qualifying configs pick
the one with median R closest to 0.5; FREEZE. Same regime for all
coordinate views (they share trajectories). If NO config lands in
the band, Exp-2b fails at setup (honest — the substrate cannot be
made to lock under event drive).

## CHANGE 2 — TIME-AVERAGED coordinate views (frozen)
After a washout of the first 20% of time-bins, average over the
remaining trajectory:
- `R_state_avg = mean_t cos(theta_i)`                       (dim N)
- `R_rel_avg   = mean_t [cos(theta_i-theta_j),
                          sin(theta_i-theta_j)]` over the SAME N/2
  fixed pairs (dim N)
- `R_events`   = static spatial event-rate/tile (unchanged; dim N)
- `A+R` via the frozen PCA->N matched-dimension rule.
Recording-time-mean Kuramoto R is DIAGNOSTIC only, not a classifier
feature. (Time-averaging is the v0-stable operation; it is applied
identically to state and relational views, so it cannot privilege
either.)

## Order (unchanged discipline)
1. Freeze (this doc). 2. g_rec=0 invariant + LOCKING op-point
selection (skill-neutral). 3. CONVERGENCE GATE first: dt vs dt/2
(nbin vs 2*nbin), per-seed `|Delta acc| < 0.03` on the PRIMARY
`R_rel_avg` (and report `R_state_avg`), assessed only where acc is
materially above chance. 4. ONLY if it converges: CLASSIFY (10-way
logistic, F5 rules, matched dim), report `R_state_avg | R_events |
R_rel_avg | A+R`, topology seed = unit, every seed.

## Pre-committed interpretation tree
- `R_rel_avg FAILS convergence` -> time-averaging did NOT rescue it
  -> the relational bridge is well-posed ONLY for constant-drive
  locking observables, not event-driven N-MNIST. CLOSE the bridge.
- `R_rel_avg converges AND >= R_state_avg` (matched dim) ->
  information migrates from cell STATES into RELATIONSHIPS. M14
  thesis SUPPORTED on well-posed footing.
- `R_rel_avg converges but < R_state_avg` -> relational not better
  than absolute here; thesis not supported.
- SEPARATELY (honest, not part of the primary claim): compare the
  phase views to the STATIC spatial baseline `R_events` (~0.71). If
  both phase views << R_events, the dynamical substrate adds little
  over spatial location on N-MNIST — reported as such, and does NOT
  change the state-vs-relational coordinate verdict.

## Discipline
Op-point skill-neutral; pairs, PCA basis, kernel, washout, grids
frozen before results; NO tuning on test; NO further rescue after
Exp-2b. Statistical unit = reservoir seed; report every seed.

## Status
Draft, frozen. Convergence gate runs before any classification
number is read.

---
# EXP-2b RESULT (2026-09-03) — time-averaged relational readout of a
# LOCKING net ALSO fails convergence. Relational bridge CLOSED.

Op-point selection (skill-neutral) DID find locking regimes:
  g_rec=2 gain{.5,1,2}: medR 0.45/0.39/0.35 (band)
  g_rec=4 gain2: medR 0.508 (band, ~0.5) <- FROZEN
  g_rec=4 gain{.5,1}: 0.90/0.72 ; g_rec=8: 0.98/0.92/0.76
So the substrate CAN lock under event drive. Invariant exact (0).

CONVERGENCE GATE (time-averaged, locking g_rec=4 gain=2):
  R_state_avg  acc 0.207/0.307  |Δ|=0.100  UNSTABLE
  R_rel_avg    acc 0.292/0.409  |Δ|=0.117  UNSTABLE
Accuracy still CLIMBS with finer dt; time-averaging did NOT rescue
convergence. Per the pre-committed tree -> CLOSE the relational
bridge on event-driven N-MNIST. No rescue after 2b.

DEEP DIAGNOSIS (sharpens M14). v0's relational robustness (corr
0.997) was for a CONSTANT-driven AUTONOMOUS locking network — no
input perturbation, so time-averaged relative phases settle and are
dt-stable. N-MNIST is EVENT-DRIVEN: the saccade events ARE the
perturbation, delivered throughout the recording, and their
integration is resolution-dependent. So even a locked, time-
averaged network inherits dt-fragility FROM THE INPUT EVENTS. The
relational-coordinate robustness does NOT transfer from autonomous
collective dynamics to input-event-driven temporal computation
under a common discretized interface. This is the M13.5 lesson at a
new level: the fragile quantity is tied to input event timing, and
no downstream relational/time-averaging operation removes it.

STANDING RESULT. On N-MNIST the STATIC spatial event-rate histogram
(R_events, 0.711) is the stable, dominant readout; every phase-
DYNAMICS view (absolute or relational, final-state or time-
averaged, weak or locking coupling) fails step-halving. The M14
"computation in relationships" thesis is NOT supported on real
event data. Robust M14 findings are methodological: (i) no
temporal-geometry resonance (Exp-1, both datasets); (ii) memory-
horizon reorders substrates (Exp-1 784 vs 64); (iii) input-event-
driven phase readouts are discretization-fragile and relational/
time-averaging does not fix it (Exp-2/2b).
