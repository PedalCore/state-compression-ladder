# M13.5 — Information Processing Capacity per hardware cost

Question (the programme's founding hypothesis, made quantitative):
does richer neuron dynamics buy computation per unit hardware?
Metric: Dambre Information Processing Capacity (linear memory +
nonlinear orders 2/3/4) + NARMA, DIVIDED BY state bits / MACs /
estimated gates. All neurons FROZEN (no primitive retraining —
M13 closed); only a linear readout trains. One shared reservoir
topology, matched conditions.

Arms (equal reservoir size, equal input drive, linear readout):
  LIF-1        (1 state, ~handful params)
  AdEx-2       (2 states)
  M13-field    (the frozen A0 vector field, 4-state)
  M13-kc1      (frozen field + 1-scalar trust corrector)
Timescale-spectrum control (the deep question — is it state COUNT
or the SPECTRUM those states carry?): a bank of leaky traces at
1 / 2 / 4 exponential timescales matched on state count.

Preregistered (before running):
P1: nonlinear IPC (orders 2-4) per state-bit is HIGHER for the
    HH-derived cells than LIF at matched cost.
P2: the timescale-bank control captures much of any advantage ->
    the resource is the timescale SPECTRUM, not HH geometry per se
    (the honest deflation to watch for).
P3: NARMA tracks IPC ordering.
Standing scope: measurement only; no primitive retraining; M14
(four-neuron re-taskability, synchronisation) waits on this
showing a per-cost advantage exists.

## v1 harness FLAWED (2026-09-03) — measured isolated single cells

v1 drove one cell of each type with the raw input and measured
IPC over its k states. Result: near-zero nonlinear capacity for
ALL cells (incl. nonlinear hh2), NARMA NaN (recursion diverged).
Not a finding — a design error: IPC is a RESERVOIR measurement;
a single deterministic unit has no population heterogeneity for a
linear readout to exploit, so nonlinear terms vanish. v1 numbers
DISCARDED. Fix (v2): reservoir of N heterogeneous cells per type
(random input weights, random biases/thresholds/timescale
jitter), IPC per TOTAL state count; NARMA input rescaled +
clipped to prevent divergence. This is the standard Dambre setup
and the honest apples-to-apples "computation per hardware" test.

## v2 SECOND confound (2026-09-03) — input-encoding mismatch

v2 fixed NARMA (no NaN; ~0.73 all cells) and reservoir structure,
but the trace banks alone got a tanh(win*x+bias) input front-end
while LIF/AdEx/hh2 got LINEAR drive. So traces' large nonlinear
IPC (C2/C3 ~0.8-1.85) measures the INPUT tanh, not dynamics; the
others show ~0 nonlinear capacity on linear drive. v2 cross-cell
numbers DISCARDED. Lesson: IPC comparison must fix the input
encoding identically across arms so nonlinear capacity is
attributable to DYNAMICS alone. This is a genuine measurement-
design problem (two confounds in two passes) — M13.5 is a real
project needing preregistered harness controls, not fast
iteration. PAUSING for harness design review rather than spawning
v3 blind — the same discipline that kept M13 solid.
Open design questions for v3: (a) identical linear input drive to
all cells (then nonlinear IPC can only come from dynamics —
expected: linear traces -> 0 nonlinear, by construction; spiking
reset + quadratic -> the real test); (b) hardware-cost proxy
beyond raw state count (spiking cells are cheaper per state than
continuous); (c) is near-zero nonlinear IPC for weakly-driven
LIF real or a drive-amplitude artifact.

## v3 QUALIFICATION (2026-09-03) — harness runs; integrity gates
## pass; SPIKING-SUBSTRATE observability gate FAILS. HALT for
## reviewer ruling before any production run.

Built hh_ipc.py to the frozen preregistration: reservoir of N
frozen heterogeneous cells, one shared W + input mask per seed,
strictly-affine input path (I = ibias + iscale*(I0 + g_in*b*u +
g_rec*W@y)), one scalar observable/cell, deterministic Legendre
target library (tau>=1, tau=0 excluded), 20-shift circular null
threshold, argmax-C operating-point search on a separate
calibration sequence. Three qualification arms: linear-trace |
LIF | HH-4state.

INTEGRITY GATES PASS (the ones that guard against a rigged
harness):
- G1 linear-trace nonlinear IPC = 0.007 ~= null. The affine input
  path manufactures NO nonlinearity. This was the whole point of
  the four non-negotiables and it holds.
- G2 C_{<=4,L} <= N for every arm (trace 4.29, LIF 3.27 at N=40).
- LIF is step-halving STABLE (nonlin ~0.2-0.4, activity ~0.13
  flat across sub=1..10) — a well-behaved reference.

GATE 3 (step-halving must not reorder arms) FAILS for the HH /
spiking substrate, and the failure is STRUCTURAL, not a bug:
  - Instantaneous membrane-snapshot observable, coarse sampling
    (0.5ms): HH IPC never dt-converges. C1 climbs 0.08 -> 0.38 ->
    0.62 -> 0.80 as dt 0.02 -> 0.0025ms; still rising. A single
    snapshot of a ~1ms spike is integration-phase-sensitive.
  - Fine sampling (0.1-0.2ms): now dt-converged, but IPC ~= 0 for
    ALL orders. iid input at 5-10kHz is far above the membrane
    cutoff; HH low-passes it to nothing. dt-stable AND dead.
  - Low-pass observable (tau_f=2ms) at 0.5ms sampling: partly
    rescues dt-stability (C1 0.30 -> 0.38 -> 0.43, still drifting)
    but SMOOTHS AWAY the spike nonlinearity -> nonlinear IPC ~= 0.
    Stable-ish but measures a linear filter, defeating the point.

DIAGNOSIS. Leaky integrators (trace, LIF-subthreshold) pass
step-halving trivially because they have no stiff event to alias.
A stiff spiker cannot be BOTH (i) sampled as one instantaneous
scalar at a fixed shared interval AND (ii) step-halving-stable AND
(iii) still expose its nonlinear (spike) computation. Any two are
achievable; not all three at once. This is a genuine ill-
posedness in the preregistered observable for spiking substrates,
found by qualification BEFORE any headline number was trusted.
The earlier smoke pass "HH nonlin=10.1" was pure coarse-Euler
artifact (sub=1, 85% of steps were blow-up/clipping) — exactly
what gate 3 exists to catch. Discarded.

Note the operating-point objective also parked LIF subthreshold:
argmax C_{<=4,L} is C1-dominated, so it rewards the linear-memory
regime, not LIF's spiking (nonlinear) regime. Second reason the
current spec under-measures event nonlinearity.

WHY I AM NOT TUNING PAST THIS. Choosing an observer/timescale
that makes HH look good is precisely the harness-design discretion
the reviewer required frozen up front. Each candidate resolution
below changes what "IPC of HH" MEANS, so it is a preregistration
decision, not a mid-run knob.

CANDIDATE RESOLUTIONS (for the reviewer to rule on before v3-prod):
1. Input-hold / masking: input piecewise-constant over tau_in
   matched to substrate timescale (e.g. 2-5ms), reservoir sampled
   once per hold; IPC delays in hold-units. Standard neuromorphic
   reservoir practice. Must still fix the observable's dt-
   stability.
2. Windowed observable, preregistered identically for ALL arms
   and counted as cost: per-hold mean(V) or spike-count, not an
   instantaneous snapshot. Fair only if EVERY arm gets the same
   window (trace/LIF too), else it is the v2 asymmetric-front-end
   mistake again.
3. Operating-point objective: select on nonlinear capacity
   C_{2:4}, or report the whole gain map and compare arms at each
   cell's own C-max, so no arm is forced subthreshold by a linear-
   dominated objective.
4. Accept that step-halving stability of a stiff spiker requires
   a stated substep floor, and fold that substep count into the
   B1 cost ledger as a real (large) HH cost — which is itself part
   of the "IPC per hardware cost" answer, not a nuisance.

STATUS: v3 harness code committed. Integrity gates green. NO
production IPC numbers exist or should be quoted. M13.5-A does not
pass qualification until the spiking-observability question is
resolved in writing. Sending back to the reviewer.
