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

DIAGNOSIS (softened per reviewer II.5). Leaky integrators (trace,
LIF-subthreshold) pass step-halving trivially because they have no
stiff event to alias. The precise, established finding is NOT a
universal impossibility: it is that UNDER THE INSTANTANEOUS-
SNAPSHOT PROTOCOL AND THE TESTED INTEGRATION SCHEME, the HH-like
substrate did not yield an observable that was simultaneously
numerically converged AND informative at the shared sampling
interval. The general lesson is the real content:
`equal sample clock != equal observation of dynamical computation`
when primitives have very different dynamical bandwidths. Snapshot
IPC is not a neutral measurement interface across bandwidths.
Found by qualification BEFORE any headline number was trusted.
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

## v3 + AMENDMENT II QUALIFICATION (2026-09-03) — HH substrate
## FAILS qualification robustly under the fixed protocol. Not a
## config artifact. Constructive path proposed; halt for reviewer.

Rebuilt hh_ipc.py to Amendment II: symbol-held common input,
LINEAR window-average observable, hold chosen by convergence (not
IPC), two operating points, physical-time cost ledger.

II.1 hold selection (dt vs dt/2, NO IPC), per-symbol window-stat
rel-RMS (must be <2%):
  Th=0.5ms  trace 2.0%  LIF 1.5%  HH 95.9% (spk17%)
  Th=1.0ms  trace 1.4%  LIF 1.1%  HH 84.9% (spk 8%)
  Th=2.0ms  trace 0.9%  LIF 0.7%  HH 71.1% (spk 5%)
  Th=5.0ms  trace 0.5%  LIF 0.4%  HH 48.5% (spk 3%)
No common hold converges -> qualification fails as specified.

Decomposition (the important part):
- HH AGGREGATE integration DOES converge: pooled window mean
  isolated (g_rec=0) 0.0632->0.0623->0.0624->0.0627, dt 0.02 ->
  0.0025 (delta ~1e-4). The integrator is fine.
- HH PER-SYMBOL window stat does NOT converge: rel-RMS 45.7% at
  g_rec=0 (ISOLATED neurons — no network chaos), rising to
  85-92% at g_rec 0.3-0.5 (past edge of chaos).
- Gate 4 (IPC ESTIMATE stability) at echo-state g_rec=0.1,
  Th=1ms: C_total 0.01 -> 0.01 -> 0.07 across dt 0.02->0.005 —
  near-NULL and NOT stable.

MECHANISM. Near threshold, whether a cell fires INSIDE a given
window is a boundary decision, and one spike is a large fraction
of a 1ms window mean; tiny dt shifts flip spikes between windows.
Those flips cancel in the global mean (aggregate converges) but
scramble the per-window fine structure the IPC readout consumes.
The linear window-mean observable does NOT rescue this — the
events still land in dt-dependent windows. For a spiking
substrate the feature carrying the nonlinear computation (spike
timing) IS the feature that fails step-halving. Only the chaotic
strongly-coupled regime yields large window variance, and there it
is maximally dt-unstable. Robust across observers, holds, coupling.

INTERPRETATION (methodological, and it matters). PASSIVE reservoir
IPC — freeze the dynamics, train only a linear readout — is well-
posed for SMOOTH / rate substrates (traces, subthreshold LIF) but
ill-posed for EVENT-based stiff spikers: their informative
dimension is not a step-halving-stable linear observable. This is
NOT the same as saying such substrates cannot compute; M13 itself
measured a spiking teacher successfully — but by TRAINING a
corrector on the dynamics (active, closed-loop, well-posed), not
by passive linear readout. So: equal sample clock != equal
observation (round 1), AND passive-linear-readout != a neutral
capacity probe for event substrates (round 2).

CONSTRUCTIVE PROPOSALS (reviewer to choose; do NOT self-select):
(P1) The fair "M13 arm" is NOT raw spiking HH — it is the FROZEN
     LEARNED SMOOTH FIELD (+corrector). That field is an MLP-
     smooth vector field and may be step-halving stable where raw
     HH is not. Testable: run the frozen comp_stage0 field as the
     reservoir cell and re-check gates 3/4. If the compressed M13
     model is MORE measurable than the substrate it approximates,
     that is itself a clean, publishable asymmetry.
(P2) Replace PASSIVE IPC with TASK-based capacity: train a small
     readout/controller on a temporal task (NARMA / delayed
     parity) per substrate, measure accuracy per hardware cost.
     Task performance is an expectation robust to spike jitter,
     sidestepping the per-window fragility while still answering
     "computation per cost."
(P3) Restrict M13.5's IPC comparison to smooth substrates
     (traces, M13 field, GRU/SSM readouts) and drop raw spiking HH
     from the IPC table, documenting WHY (this finding) rather
     than quietly.

STATUS: harness code committed; integrity gates (trace-null,
C<=N) green; HH substrate fails gates 3 AND 4 robustly. No
production IPC numbers. M13.5-A remains unqualified pending the
reviewer's choice among P1/P2/P3.
