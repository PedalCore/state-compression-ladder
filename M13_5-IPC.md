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

## P1 RESULT (2026-09-03) — the FROZEN M13 primitive ALSO fails
## passive-IPC convergence. Diagnostic settled. Pre-registered
## tree -> branch 3 -> P2 (task-based). Not a config artifact.

Ran the EXACT deployed M13 primitives as reservoir cells (no
retraining, no adapter change, no IPC-friendly tweak): the frozen
learned field comp_stage0_s0 (diagnostic) and the canonical
kc=1 = field + bounded trust corrector (hyb-rec1-trust-sel-v2_s0,
audit-fixed). Same symbol-held input, affine path, W, one-scalar
V observable, linear window mean, target library. Gate II.1 halves
the FIELD substep (SUB 10 vs 20); the corrector's 0.1ms cadence is
model definition, held fixed.

Gate II.1 (window-mean rel-RMS, NO IPC), g_rec = 0 / 0.1 / 0.3:
  M13-field  44.0% / 61.1% / 81.5%
  M13-kc1    54.6% / 65.6% / 88.2%
Gate 4 (IPC estimate stability, Th=1ms, echo-state g_rec=0.1):
  M13-field  tot 0.12 -> 0.47 (sub 10->20), NL = 0.00
  M13-kc1    tot 0.08 -> 0.00 (sub 10->20), NL = 0.00

BOTH fail convergence, ~as badly as raw analytic HH (44% at
g_rec=0 field vs 45.7% HH). The bounded corrector does NOT
regularize — it is marginally WORSE (adds a small discrete
per-record perturbation to near-threshold cells). Gate 4 IPC is
near-null AND unstable for both.

DIAGNOSTIC (the reviewer's field-vs-field+delta question, ANSWERED):
stability does not come from the field approximation, and the
corrector does not add it. The learned surrogate was trained to
REPRODUCE spikes, so it reproduces the threshold-event structure
that makes per-window spike assignment dt-fragile. MLP smoothness
is irrelevant: the trajectory still has near-threshold
branchpoints. Compression preserved behaviour WITHOUT changing
numerical/observational accessibility under passive linear-readout
IPC. (So NOT the "better-conditioned coordinates" result.)

NARROW, DEFENSIBLE METHODOLOGICAL FINDING (final wording):
Under a passive reservoir-IPC protocol with fixed symbol windows
and a linear scalar observation, neither analytic HH nor the
deployed M13 surrogate admits a per-symbol feature representation
that is simultaneously step-halving-converged and informative in
the spiking regime. Aggregate integration converges; the
instability is localized to event assignment across observation
windows, and it is intrinsic to spike-producing dynamics (learned
or biological), not to a particular integrator or to the HH
equations specifically.

DECISION (pre-registered interpretation tree, branch 3): STOP
forcing spiking substrates into passive IPC. Move to P2 =
task-based temporal-capacity benchmarks (accuracy on NARMA /
delayed-parity per hardware cost), whose scores are expectations
robust to spike-time jitter. Retain P3 as documentation: passive
linear-readout IPC is kept for SMOOTH / rate substrates (trace
banks, and any smooth readout models); event-based spikers are
EXCLUDED from the passive-IPC table because the observable is not
numerically well-posed under the common protocol — with THIS
result as the cited reason, not a silent omission. P2 is a new
protocol and requires its own preregistration + reviewer sign-off
before any run.

## P2 CONVERGENCE GATE (2026-09-03) — DIRECTION confirmed;
## M13 result BORDERLINE, not a clean pass. Halt for reviewer.

Ran the frozen F4 minimal gate (trace|LIF|M13-kc1, d=2,8, delayed
recall + 2-bit parity, substeps n vs 2n, op=(0,1,0.1), Th=1ms,
4 paired topology seeds, frozen F5 readouts, ±1 shared drive).

Task-score deltas (|dR2| recall, |dAcc| parity):
  trace   mean dR2 0.000-0.003, worst 0.004; dAcc <=0.011.  clean.
  LIF     mean dR2 0.000-0.003, worst 0.004; dAcc <=0.005.  clean.
  M13-kc1 mean dR2 0.011 (d2)/0.021 (d8), WORST dR2 0.040 (d8);
          mean dAcc 0.019 (d2)/0.010 (d8), worst dAcc 0.030 (d2).

READ (honest; the auto PASS=True checked only the mean and is too
lenient vs F3+F6):
+ DIRECTION CONFIRMED: task scores are 10-40x more convergent than
  P1 feature-level (44-88% RMS -> ~1-4% abs). Functional output is
  far more stable than microscopic event placement, as P2 bet.
- NOT a clean pass for M13: convergence ~1 order looser than the
  smooth arms; seed2 d=8 dR2=0.040 BREACHES the 0.03 tolerance
  (the catastrophic-seed case F6 says to weigh); seed0 d=2
  dAcc=0.030 on the line.
- SYSTEMATIC, not noise: d=8 recall R2 drops sub10->sub20 on every
  seed (0.053->0.024, 0.060->0.020, 0.018->0.011). Halving dt
  consistently reduces M13's long-delay skill; the score is
  attenuating like the features, heavily damped but not converged.
- UNDERPOWERED: at the frozen representative point M13 skill is
  near-null (recall R2<=0.12, one negative; parity ~chance), so a
  0.04 delta is ~80% relative. The gate can barely exercise M13
  here; the representative point may simply be a poor M13 operating
  point (weak echo-state / not usefully spiking).

VERDICT: borderline. The task-based reframing clearly helps
(orders of magnitude), but M13's score is not yet demonstrably
step-converged at long delay, and the test is confounded by
near-null skill at the fixed point. Fix (code): PASS logic must
enforce F3 per-seed, not mean-only. Decision needed from reviewer:
(a) re-run the gate at an M13 operating point with non-trivial
skill selected by the FROZEN calibration rule (not hunted), with
more seeds, to test convergence where skill exists; or (b) accept
borderline and carry the looseness as a documented caveat on all
M13 numbers; or (c) if long-delay score drift persists at a
skillful point, conclude P2 also fails for M13 at long delay ->
P3. NOT advancing to the full ladder until this is resolved.

## P2 QUALIFICATION = FAIL (2026-09-03, reviewer ruling). Positive
## attenuation result banked. M13.5 at a strategic fork.

The automated mean-based PASS=True is SUPERSEDED. Under the frozen
per-seed criterion (F3+F6), M13-kc1 did NOT meet the gate:
- one topology (seed2) exceeded 0.03 at d=8 (dR2=0.040);
- another (seed0) lay on the boundary (d=2 dAcc=0.030);
- long-delay recall decreased SYSTEMATICALLY under step halving
  (0.053->0.024, 0.060->0.020, 0.018->0.011) — not jitter around a
  stable value; apparent long-memory skill partly disappears as the
  numerical solution is refined;
- the qualification operating point produced near-null skill, so the
  gate is a weak functional-stability test (0.05->0.02 is tiny in
  absolute terms but huge relative to the useful computation).
Code fixed: PASS now enforces per-seed worst<0.03, not mean.

RESULT-LOG WORDING (reviewer): "Task-level readouts strongly
attenuated the discretization sensitivity observed in per-window
spiking features, reducing changes by roughly one to two orders of
magnitude. However, the canonical M13-kc1 arm did not satisfy the
preregistered per-seed convergence criterion: one topology exceeded
the 0.03 tolerance, another lay on the boundary, and long-delay
recall decreased systematically under step halving. Because the
qualification operating point produced near-null task skill, the
experiment establishes attenuation of microscopic timing
sensitivity but does not establish convergence of useful
computation."

TWO HYPOTHESES, now separated:
- P1 FALSIFIED: "the compressed field regularizes away event
  sensitivity." It does not.
- P2 PARTIALLY SUPPORTED: "functional decoding can suppress event
  sensitivity." Yes, strongly (1-2 orders of magnitude).
- P2 NOT ESTABLISHED: "useful decoded computation is
  discretization-stable." Open.

METHODOLOGICAL RESULTS BANKED (publishable as-is):
1. Event-faithful spiking dynamics are hard to rank with passive
   common-clock capacity probes.
2. Task decoding suppresses much of that sensitivity, but not
   enough to certify stable USEFUL computation under the tested
   protocol.

FORK (reviewer): either (P2b) ONE final preregistered experiment —
select each arm's operating point by COARSE-STEP validation skill
under an equal fixed search budget, REQUIRE a minimum useful skill
(e.g. recall R2>0.2 or parity materially above chance) BEFORE
convergence is assessed, freeze, then test n vs 2n on untouched
data (asks: when the primitive is actually computing usefully, is
that computation stable to refinement?) — labelled a NEW prereg,
NOT a repair, and the LAST M13.5 attempt; OR close M13.5 now
without a hardware-advantage claim and move to M14. Reviewer leans
CLOSE-AND-M14 unless proving M13's per-cost advantage is
strategically essential. Note the failure points AT M14: the
fragile quantity is t_spike; the stable quantity is the downstream
relational one (task decision -> phase/sync/cluster regimes), which
is exactly what M14/ONN measures. No further rescue tree after P2b.

## P2b RESULT (2026-09-03) — M13-kc1 reaches min-skill at NO
## equal-budget operating point. M13.5 CLOSES, no hardware-
## advantage claim. Decision-tree branch 3. No rescue after this.

Skill-selected equal-budget operating points (18-config grid,
selection on validation S_recall only, coarse step), then min-skill
gate, then convergence on 6 HELD-OUT topology seeds:
  trace   op=(0.0,2.0,0.3)  val R2(d2)=0.224  min-skill=YES  ->
          convergence PASS per-seed (worst dR2 0.004, dAcc 0.005).
  LIF     op=(0.0,0.5,0.3)  val R2(d2)=0.227  min-skill=YES  ->
          convergence PASS per-seed (worst dR2 0.004, dAcc 0.005).
  M13-kc1 op=(0.0,1.0,0.3)  val R2(d2)=0.110  parity acc=0.544
          -> min-skill=NO (need R2>0.2 OR parity>0.55). Convergence
          NOT assessed (honest, per prereg).

M13-kc1's SKILL-MAXIMIZING point across the whole equal-budget grid
reaches only recall R2=0.11 (half of trace/LIF's 0.22) and parity
0.544 (below the 0.55 gate). At equal cell count it delivers LESS
useful temporal skill than a leaky integrator, and below the useful
threshold — so there is no useful computation whose convergence
could be certified. Pre-committed branch 3: cannot certify -> M13.5
closes with NO hardware-advantage claim. Reviewer's rule: no rescue
tree after P2b. This is final.

SCOPE OF THE NEGATIVE (stated precisely, not overclaimed): as a
FROZEN reservoir cell under a passive common-clock linear-readout
protocol with affine input, M13-kc1 does not provide more useful
temporal task skill than LIF at equal budget. This does NOT
contradict the M13 paper (M13 was validated as a TRAINED closed-
loop corrector reproducing HH spike trains at F1>0.9) — it says the
founding M13.5 hypothesis ("richer neuron dynamics buy more
computation per hardware cost") is NOT supported by this
measurement of the frozen primitive as a passive reservoir. A
charitable "M13 needs more cells" would mean MORE hardware for LESS
skill — the opposite of a per-cost advantage.

M13.5 FINAL LEDGER (three banked results, one closed hypothesis):
+ P1: M13 faithfully preserves spike-event geometry -> falsifies
  "compression regularizes observability."
+ P2: task decoding attenuates event-timing sensitivity 1-2 orders
  vs passive features (a real, hardware-relevant robustness).
+ Methodological: event-faithful stiff spikers are hard to rank
  with passive common-clock capacity probes; and even skill-gated
  task decoding does not certify a per-cost advantage for the
  frozen M13 primitive.
- Hypothesis "richer dynamics buy computation per cost": NOT
  supported under this measurement. No claim made.

NEXT: M14. The fragile quantity throughout was t_spike; the stable,
useful quantities are relational/downstream (task decision ->
phase relationships, synchronisation, cluster membership,
integrated collective regimes) — exactly M14/ONN's object, which is
intrinsically less dependent on which side of a 0.5ms window a
spike landed. The M13.5 failure points directly at M14.
