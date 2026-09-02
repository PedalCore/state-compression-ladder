# M13 — the state-compression ladder: how many dynamical degrees
# of freedom does a neuron's computation actually need?

Origin: collaborator proposal (2026-09-01) — approximate the
*computationally relevant* state of biological units with compact
learned dynamical systems, compose them, allow heterogeneous
timescales. Sharpened here into the program's form: a measured
ladder with a known-answer calibration rung.

The program already owns pieces of this question: WKV = 2 counters,
longhorn = 1 diagonal memory/channel, M10's clock law (17 params of
phase beat every oscillator bank), M12's five-instances law
("conflicting temporal configurations demand independent state
variables; parameter search cannot substitute for state"). M13 asks
the question at the single-unit level, quantitatively.

## Design (rung 1 — calibration on known dimension)

Teacher: standard squid-axon Hodgkin–Huxley (V, m, h, n — true
dynamical dimension 4). Internal integration dt=0.01 ms
(exponential Euler on gates), recorded at 0.1 ms. Drive:
Ornstein-Uhlenbeck current (per-sequence mean 0-8, sigma 0.5-4,
tau 3-50 ms) plus 0-3 random step pulses (-3 to +12, 20-200 ms),
clipped to [-5, 20] uA/cm^2 — spans silent, fluctuation-driven and
mean-driven regimes. 256 train / 32 val / 32 test sequences of
1000 ms.

Surrogates: k-state GRU cell (input = scaled current, hidden = k,
linear readout to normalized voltage), k in {1, 2, 4, 8}. TBPTT
chunks of 1000 steps (100 ms), Adam, spike-region samples
(V > -20 mV) weighted 4x in the MSE. Sub-1k parameters at k=8 —
these are units, not networks.

Metrics:
1. voltage RMSE (mV, test);
2. spike-timing F1, +-2 ms tolerance (spikes = upward 0 mV
   crossings, 2 ms dedup, same detector on teacher and surrogate);
3. behavioral signatures OUT OF DISTRIBUTION of the training
   drive: f-I curve (HH is type II — discontinuous onset near
   ~50 Hz at rheobase ~6.2), and the anodal-break rebound spike
   (release from I=-3 hyperpolarization fires with NO positive
   drive).

## Preregistered predictions (written before any training)

P1. Both fit metrics improve sharply k=1 -> 2 -> 4 and saturate at
    k=4: the ladder should recover the teacher's true dimension
    (k=8 ~ k=4 within noise).
P2. k=1 fails qualitatively, not just quantitatively: no rebound
    spike, distorted f-I onset (type II onset needs the
    subthreshold resonance a single state cannot express). A
    1-state unit is a committer, not a neuron (cf. M12).
P3. k=2 captures most of the spike-timing F1 (the Izhikevich
    claim, learned rather than hand-derived) but underfits
    voltage shape and the rebound amplitude.
P4. The k=4 surrogate at 0.1 ms is a ~10x cheaper-per-step drop-in
    for the 0.01 ms teacher (and needs no gating lookups) — the
    "compact digital approximation" quantified.

## Later rungs (preregistered direction, not yet run)

R2: hand-designed classical arms fitted to the same data by CMA-ES
    (LIF-1, Izhikevich-2, AdEx-2) vs the learned ladder at equal
    state count — does learning beat 50 years of hand design at
    matching a *specific* teacher?
R3: adapting teacher (HH + slow M-current, dimension 5) — does the
    ladder's saturation point track the added slow variable? This
    is the instrument test: detect unknown effective dimension.
R4: composition — a ring/chain of k-state surrogates vs one
    (nk)-state monolith at matched total state: where does
    locality + heterogeneity beat lumping (the collaborator's
    mixture-of-dynamics idea, and M12's five-instances law run
    forward).
R5: task rung — budget-matched sequence tasks (our M9/M10 suites)
    with 1/2/4/8-state units: which states pay per byte on *tasks*
    rather than on imitation.

Honesty notes: rung-1 saturation at k=4 validates the method, not
the hypothesis that richer units help on tasks (that is R5). A
GRU's gates give it timescale flexibility a plain RNN lacks; the
ladder measures state COUNT, with cell class held fixed.

## Run 1 (2026-09-01): GATE FAILED — recipe, not dimension, was measured

Instrument gate (implicit until now, explicit from here on): the
ladder is only interpretable if the LARGEST k fits the teacher well
(target: spike F1 > 0.9, RMSE < 5 mV). Run 1, 12 epochs, linear
readout, pure weighted-MSE: k=1 RMSE 17.9/F1 0.00, k=2 18.8/0.00,
k=4 15.0/0.51, k=8 20.2/0.23. Non-monotone in k, no surrogate
fires on constant drive (all f-I RMSE = 53 Hz = the never-spikes
score), zero rebounds. MSE alone prefers blurred spikes; the
optimizer, not the state count, is binding. No prediction can be
scored from this run.

Recipe v2 (declared before running): memoryless MLP readouts
(k->32->1; adds no state, ladder still measures k), auxiliary
spike head trained with BCE on +-0.3 ms spike indicators
(training aid; PRIMARY F1 still scored by the same 0 mV voltage-
crossing detector as the teacher), spike-region weight 10x,
30 epochs, cosine lr 3e-3 -> 3e-4.

## Related work (added 2026-09-01, user-supplied)

1. Wan, Karniadakis & Stinis, "From LIF to QIF: toward
   differentiable spiking neurons for scientific ML" (npj AI,
   2026; s44387-026-00121-2). QIF's quadratic subthreshold keeps
   spiking smooth enough for backprop — the mirror image of M12's
   HEU non-differentiability blocker. Hook -> R2/R5: add QIF as a
   1-state arm; it varies NONLINEARITY at fixed k, orthogonal to
   our k-ladder (GRU fixed). Sharpens P2: QIF onset is type-I-like;
   a 1-state unit of any nonlinearity should still fail HH's
   type II onset and rebound (both need a second, resonant state).
2. Tandale & Stoffel, "Meta-learning hybrid spiking networks as
   physics-based nonlinear solvers" (npj Unconventional Computing,
   2026; s44335-025-00048-y). LIF-gated graded outputs (spike
   mechanism as learned selectivity/regularization), Loihi-2
   deployment. Precedent that spiking dynamics pay as parameter
   efficiency in trained solvers; adjacent to our rwkv-spiking
   line more than to rung 1.
3. Freddi et al., "Mean-field criticality in spiking networks for
   reservoir computing" (Sci Reports, 2025; s41598-025-18004-y).
   Closed-form critical coupling <W>_crit for LIF reservoirs —
   principled fixed-physics initialization, no tuning. Hook -> R4:
   when composing many k-state units, their formula is a candidate
   operating point; test edge-of-chaos vs off-critical composition
   with our stability metrics (M10-style fixed dynamics + trained
   readouts is exactly their regime).

## Run 2 (2026-09-01): GATE FAILED AGAIN — optimization variance
## dominates; declaring recipe v3 (teacher forcing)

v2 results: k=1 RMSE 25.7/F1 0.00, k=2 25.4/0.00, k=4 23.2/0.00
(WORSE than its own v1 run, 0.51 — same recipe, different seed
behavior), k=8 17.7/0.67. Still no constant-drive firing, no
rebound. Diagnosis: free-run regression onto sharp rare events —
mistimed spikes cost double (miss + false alarm), so the optimizer
suppresses spiking; run-to-run variance confirms a hard loss
landscape, not a capacity limit.

Recipe v3 (declared before running): autoregressive observable
feedback with scheduled sampling — input becomes [I_t, v_{t-1}],
where v_{t-1} is the TEACHER's voltage with probability 1-eps and
the model's own (detached) prediction otherwise; eps ramps 0 -> 1
over the first 60% of epochs; evaluation is ALWAYS full free-run
from rest. This is standard teacher forcing from the dynamical-
system-reconstruction literature, and the same fix (scheduled
sampling) that repaired M10-P calibration.

HONEST STATE ACCOUNTING: the fed-back voltage is itself a state
variable. Surrogate total state = k + 1. The ladder therefore
sweeps k in {1,2,3,4,8} (total 2,3,4,5,9), and P1 is restated:
saturation at TOTAL state 4, i.e. k=3. All other predictions
carry over with k read as total-state minus one.

## R2 declared in detail (2026-09-01, before running; v3 still going)

v3 interim: k=1,2,3 all F1 0.00 free-run (the eps -> 1 anneal
collapses every arm to silence so far). Whatever k=8 does, R2 runs
next: classical hand-designed units fitted to the SAME train
sequences by CMA-ES (loss = (1 - F1_2ms) + 0.05 * subthreshold
RMSE; 16 fitting seqs, 4 restarts), scored on the same test set +
signatures. Arms: LIF (1 state + reset), Izhikevich (2), AdEx (2).
Spikes for classical arms are their explicit reset events (their
mechanism, honestly theirs); voltage RMSE via affine map, reported
but secondary.

R2 predictions:
R2a. Fitted Izhikevich-2 beats every learned arm's current F1 by a
     wide margin — the bottleneck in runs 1-3 is TRAINABILITY of
     spiking dynamics under gradient descent, not state count.
R2b. LIF shows zero rebound spikes STRUCTURALLY (passive leak
     cannot overshoot on release); at least one of the 2-state
     arms recovers the anodal-break rebound.
R2c. No classical arm matches HH's type II f-I discontinuity as
     well as its overall F1 (Izhikevich fitted for timing tends
     type I unless parameters land in the resonator regime).

## Round 1 synthesis (2026-09-02): three learned recipes, one
## classical round — the honest scorecard

v3 final: ALL arms (k=1,2,3,4,8; total state 2-9) F1 0.00 in
free-run. Notably WORSE than v2 at k=8 (0.67 -> 0.00): detached
scheduled sampling taught reliance on teacher feedback, then
removed its quality with no gradient path to learn self-
correction — the documented bias of detached scheduled sampling.

R2 classical (CMA-ES, same data): LIF-1 F1 0.279 / f-I 107.5 /
rebound 0; Izhikevich-2 0.335 / 44.7 / 0; AdEx-2 0.368 / 49.0 / 0.

Prediction scoring:
- P1-P4: UNSCOREABLE — the instrument gate (largest k at F1>0.9)
  never passed in three recipes. The ladder has not yet measured
  HH's dimension.
- R2a: CONFIRMED with a caveat — every designed arm beats every
  learned arm (0.28-0.37 vs 0.00), but the designed ceiling is
  ~0.37, not mastery.
- R2b: HALF-FAILED — LIF's zero rebound is structural as
  predicted, but NEITHER 2-state arm recovered the rebound:
  timing-optimal parameters sit in the fast-spiking regime, not
  the resonator regime. Timing fit and signature fit are in
  tension at 2 states.
- R2c: CONFIRMED — no arm matches type II; Izhikevich's 44.7 is
  the only f-I score better than never-firing (53).

Two real findings survive the failed gate:
1. TRAINABILITY, not capacity, binds the learned arms: 9 total
   states + 3 recipes < 1 designed state + parameter search.
2. STATE, not design, binds the classical arms: AdEx and
   Izhikevich converge to fitting losses within 1e-4 of each
   other — a 2-state class ceiling (~F1 0.35) for fluctuation-
   driven HH timing, echoing the five-instances law from the
   designed side.

Recipe v4 (declared before running): return to the v2
configuration (no feedback input — the only arm ever to fire in
free-run), epochs 20 -> 60, ks {2,3,4,8}, seed 0, same gate.
If v4 also fails the gate, rung 1 concludes as a negative result
with the two findings above as its product, and R3+ proceed with
designed/hybrid arms.

## CORRECTIONS (2026-09-02, after collaborator review) — published,
## not patched

1. P1's framing was WRONG, not merely unscoreable. Canonical HH
   has 4 explicit state variables, but 4 is not its minimal
   observable/computational dimension: 2-D reductions (m -> m_inf(V)
   slaving, h/n collapse — Krinsky-Kokoz, Rinzel, the FitzHugh-
   Nagumo lineage) reproduce much of its behavior. The ladder
   measures the MINIMAL REALIZATION for a model class x input
   distribution x loss x observables — saturation at 2 would not
   be instrument failure, saturation at 4 would not prove
   dimension recovery. All rungs reread accordingly.
2. RETRACTED: "a 2-state class ceiling, not optimizer luck" (also
   posted publicly; correction posted to the project). Two 2-state
   families at similar losses do not bound the class of all
   2-state systems — objective, drive distribution, CMA budget,
   parameterization, and local optima are all uncontrolled.
   Replacement: two quite different canonical 2-state models
   converged to remarkably similar performance under the same
   criterion, SUGGESTING but not establishing a representational
   rather than model-specific bottleneck.
3. Downgraded: "you can't parameter-search your way out of
   missing state" is a hypothesis these results suggest (and M12
   demonstrated in ITS setting), not something demonstrated here.
4. Sharpened reading of runs 1-3: they measured capacity x
   optimization x rollout-stability x loss-geometry, entangled.
   Scheduled sampling itself has known consistency problems.
   Rather than v5/v6/v7 on the same entangled problem, redesign.

## REDESIGN — three compressions, experiment ladder A-D
## (declared before running)

A. MECHANISTIC: remove partial observability. Supervision on the
   full teacher state (V,m,h,n); model = encoder E(s_0) -> z_0,
   SAME GRU cell class as runs 1-4 (input = current only), decoder
   z -> (V,m,h,n). Implementation gate: k=8 must reach near-
   perfect rollout (V-RMSE < 2 mV, F1 > 0.95) — if it cannot,
   the problem is implementation, full stop. Then sweep
   k = 1,2,3,4,8. Expectations: k=1 fails; k=2 tracks the known
   2-D reduction (good V/spike behavior, imperfect gate
   trajectories); k >= 4 near-perfect. If A's gate passes where
   B's failed, partial observability + loss geometry — not the
   cell class — was the binding constraint of runs 1-4.
B. OBSERVABLE: current (+ voltage history) -> future voltage.
   Runs v1-v4 belong here; delay embeddings are the classical
   fix and a future arm.
C. FUNCTIONAL: behaviors only (spike timing, f-I, rebound,
   adaptation) — closest to "does extra state buy computation".
D. COMPUTATIONAL: downstream task performance per unit state
   (the R5 task rung, renamed).

The physical -> observable -> behavioral -> computational
progression is the project's actual question; rung-1's "knee at
k=4" was a special case and is retired as a headline claim.

## Experiment A interim + wording correction (2026-09-02)

Results so far (20 epochs, seed 0): k=8 F1 0.712 / V-RMSE 17.9 /
f-I 44.7 (fires under constant drive); k=4 0.468 / 21.7 / 53.0
(fires ONLY under fluctuating drive); k=3 0.000. Gate not passed.

WORDING CORRECTED (per review): the k=8 sub-gate result is "not
evidence for insufficient latent dimensionality or partial
observability" — NOT "a training/dynamics-class problem, not
capacity." With k=8 > 4 explicit teacher states and full
supervision, the unresolved bottleneck is among: optimization,
transition-class expressivity of the discrete GRU map,
discretization, rollout stability, loss scaling. "Capacity" also
includes the GRU transition's own expressive capacity.

Confounds logged before 2/1 land: (a) k=8's F1 was 0.00 through
ep9, then 0.60/0.67/0.70 — a late sudden transition, still
climbing at ep20; the 8-vs-4 gap partly reads as "larger models
cross the spiking transition earlier at fixed budget"; (b) single
seed — earlier non-monotone runs mandate multi-seed replication
before any capacity-law claim. What IS noteworthy: k=8 crossed a
QUALITATIVE boundary k=4 has not (sustained constant-drive
firing), suggesting extra state buys dynamical regime, not just
voltage error. Restrained summary: a provisional state-dependent
performance hierarchy under a fixed learned dynamics class —
not a measurement of minimal realization.

Normalization audit (per review): per-variable normalized stds
V 0.171 / m 0.211 / h 0.140 / n 0.116 — no variable dominates the
uniform state loss. Caveat stands: dynamical sensitivity is not
uniform (I_Na ~ m^3 h), so state-loss fidelity does not imply
dynamical fidelity.

## Experiment A0 declared (before running): the diagnostic fork

Above A in the ladder: NO latent, NO encoder/decoder, NO
recurrence. Plain MLP transition F(V,m,h,n,I) -> next state
(residual, 0.1 ms flow map), trained on all teacher transitions
as iid pairs. Two evaluations, separated:
1. teacher-forced ONE-STEP error — if not excellent, something
   basic is wrong (scaling/optimizer/architecture/loss);
2. AUTONOMOUS rollout from rest — recursive self-feeding.
Fork: one-step bad -> function-approximation problem; one-step
good + rollout bad -> stability/accumulated error; both good ->
the GRU formulation was the problem and the ladder rebuilds on
this transition class. Secondary mode: learn the analytic vector
field (RHS at recorded states) and integrate with Euler substeps
— separates vector-field learning from discretized transitions
(the CfC/continuous-time question, testable cheaply).

## Experiment A final + A0 fork results (2026-09-02)

A (full-state supervision, 20 ep, seed 0): k=1 F1 0.000 / k=2
0.000 / k=3 0.000 / k=4 0.468 / k=8 0.712. Monotone, with two
QUALITATIVE transitions: spiking appears between k=3 and k=4;
autonomous constant-drive firing appears between k=4 and k=8.
Restrained reading (per review): a provisional state-dependent
performance hierarchy under a fixed learned dynamics class — NOT
a minimal-realization measurement (gate unpassed; single seed;
k=8's late learning transition confounds budget with capacity).

A0 fork (no latents, no recurrence, MLP 128x128, 8 ep):
- step mode (0.1 ms flow map): one-step rel. RMSE V 0.37 / m 0.20
  / h 0.14 / n 0.06 — NOT excellent; rollout explodes (56.7 mV,
  F1 0.005). The flow map's curvature concentrates in the thin
  spike-upstroke region that iid sampling barely weights.
- deriv mode (analytic HH RHS, Euler substeps): one-step rel.
  RMSE 0.10 / 0.05 / 0.06 / 0.02 — 4x better (smoother target);
  rollout STILL fails (84 mV, F1 0.01) but differently: fires
  spuriously (4 false rebounds) and saturates the clamps rather
  than going silent. 5-10% vector-field error destroys the limit
  cycle.

Fork verdict so far: "one-step decent, rollout bad" — between the
review's branches. Before attributing to dynamical stability, the
one-step fit must be pushed to excellent (a closed-form smooth
RHS should fit to <1%).

A0b declared (before running): deriv mode, 40 epochs, width 256,
spike-region sample weighting. Question: does rollout fidelity
improve CONTINUOUSLY with vector-field precision, or is there a
precision cliff below which the limit cycle cannot be maintained?
Either answer is informative: continuous -> budget problem;
cliff -> quantifies the precision a learned dynamical
approximation of a neuron actually needs (directly relevant to
the substrate question: hand-designed dynamics carry their
qualitative regime for free; learned ones must buy it with
precision).

## A-seeds declared (2026-09-02, before running; per review):
## replicate ONLY the k=3/4 boundary

Priority over extending the sweep to k=5,6,7: rerun k=3 and k=4
at seeds 1-4 (seed 0 exists from experiment A), identical config.
Discriminates "state transition" (k=3 consistently ~0, k=4
consistently ~0.4-0.5) from "optimization probability" (both
bimodal across seeds). Also logged: the k=3->4 / HH-has-4-
variables coincidence is explicitly NOT to be connected until
seeds + budget matching + a second dynamics class reproduce it.

v4 interim reinforces the budget confound: the OBSERVABLE-track
k=2 arm reached F1 0.365 at 60 epochs (voltage-only supervision,
2 latent states, firing) — every earlier "small k cannot spike"
observation at 20-30 epochs was at least partly budget.

## A0b result (2026-09-02): the fork answers CONTINUOUS —
## and the first gate-adjacent learned model

Vector-field precision 5-10% -> 1-2% (40 ep, width 256, spike-
weighted) transformed rollout: V-RMSE 84.3 -> 11.35, F1 0.01 ->
0.864, f-I RMSE 53 -> 10.1 Hz, rebound 4-spurious -> EXACTLY 1
(correct). Three firsts: first learned model to fire the anodal-
break rebound; first to approximate the type II f-I curve; first
gate-adjacent F1. Signatures emerge in order with precision:
firing -> f-I shape -> rebound.

Fork verdict: rollout fidelity improves CONTINUOUSLY with vector-
field precision (no cliff found yet) — the earlier failures were
precision/budget, and the representation of time matters
enormously: learned vector field + integrator >> discrete GRU map
at comparable size (the continuous-time/CfC hypothesis, supported
cheaply). The surgical compression ladder (hide state -> compress
state) should rebuild on THIS transition class.

A0c declared (before running): 80 epochs, width 512, same
protocol. If the gate (F1 > 0.95 equivalent, V-RMSE continuing
down) passes, the calibration demanded by the redesign exists and
compression becomes an instrument.

## Canonical public home (2026-09-02):
## github.com/PedalCore/state-compression-ladder

M13 now has a standalone public repo (like heu-replication):
ported scripts (imports flattened, results/ layout), NOTEBOOK.md =
this file, results JSONs + logs committed, dataset/checkpoints
regenerable. Development continues here in whitebox/ while runs
are live (A0c, A-seeds, v4 execute against these paths); files
sync to the repo on each milestone. The repo link supersedes
whitebox-lm in future project posts.

## Claim tightening (2026-09-02, per review) + restructure around A0

TIGHTENED: (1) A0b does NOT show the GRU failures were "just
budget/precision" — the representation class changed between
those experiments. Defensible version: A0 rules out HH being
intrinsically too difficult for a learned approximation at this
scale; the remaining GRU failures arise from some combination of
optimization and the discrete recurrent transition
parametrization, while the vector-field formulation is a
substantially better-conditioned route to autonomous dynamics.
The representation of time may matter more than training budget.
(2) "which behaviors become REPRESENTABLE, in what order" ->
"as vector-field fidelity increases, behaviors are RECOVERED BY
THIS TRAINED APPROXIMATION in a reproducible-looking hierarchy:
firing -> f-I -> rebound." Open question (a rung of its own): is
the hierarchy intrinsic — do behaviors require systematically
different approximation fidelity — or specific to this training
distribution and objective?

RESTRUCTURE: the project skeleton is now A0 -> A -> B -> C -> D
(learn the known dynamics -> hide the coordinates -> infer state
from observables -> preserve behavior -> preserve computation).
The compression instrument INHERITS the A0 representation:
z = E(x), dz/dt = F(z, I) explicitly integrated, x_hat = D(z) —
one change only vs the positive control: the dimension of the
dynamical state. The GRU is retired as the primary compression
instrument (kept as the B-track historical baseline).

## Declared (before running): latent-field ladder + 2-D reduction
## control

hh_latentfield.py: E (4 -> k, memoryless) -> learned latent
vector field dz/dt = F(z, I) integrated with Euler substeps ->
D (k -> 4, memoryless). Sweep k = 1,2,3,4,8 AFTER A0c settles
the uncompressed positive control.

hh_reduced.py (control, no training): the classical Krinsky-
Kokoz/Rinzel-style 2-D HH reduction — m = m_inf(V), h = c - n
(c = mean(h+n) from train data), leaving state (V, n) — simulated
under the identical protocol and scored on the same metrics.
Purpose: a sanity range for latent k=2. If learned k=2 ~ the
hand-derived reduction, the instrument tracks classical model
reduction; if learned k=2 is awful while the reduction works,
then COORDINATES AND DYNAMICAL STRUCTURE matter beyond dimension
— the substrate thesis in one comparison.

## 2-D reduction control result (2026-09-02): coordinates matter,
## already

The classical reduction (m = m_inf(V), h = 0.8622 - n; slaving
std 0.035 — Krinsky-Kokoz holds well on our drive) with ZERO
fitting: F1 0.325 / f-I RMSE 26.4 / rebound EXACTLY 1 (correct).
Compare fitted-to-this-data 2-state arms: Izhikevich 0.335 / 44.7
/ 0, AdEx 0.368 / 49.0 / 0. Equal state count, same protocol:
the derived model inherits HH's coordinates, so the rebound
mechanism (h de-inactivation under hyperpolarization, preserved
as h = c - n) and half the f-I error come for free; the fitted
models bought a few F1 points by abandoning the mechanism.
Restrained claim: in this one construction, coordinates carry
behavioral signatures that data-fitting at equal dimension did
not find. The latent-field k=2 rung now has its sanity range:
match the reduction ~ instrument tracks classical model
reduction; fall far below it while the reduction stands ~
dimension is not sufficient, structure is.

## Framing consolidation (2026-09-02, per review)

Boundary wording: under this GRU regime BOTH k=3 and k=4 reach
spiking solutions; k=4 apparently more reliably and with higher
F1. The single-seed 3->4 transition was not a hard state-dimension
boundary. Report as distributions (success rate, median F1,
best-of-N), no significance claims at n=5.

The project's axis is now dimension x coordinates x dynamics x
objective — not state count alone. The latent-field k=2 question
is correspondingly: can learning DISCOVER a two-dimensional
coordinate system that preserves the same dynamical invariants a
human-derived reduction preserves?

A0c hinge, both branches declared: pass -> trusted instrument,
everything prior = calibration story. Narrow fail with fidelity
still improving -> the instrument is a smooth accuracy-to-
behavior curve, not dead.

The behavior ordering (firing -> f-I -> rebound) is preserved as
an empirical observation of THIS regime, potentially reflecting
that different dynamical signatures need different fidelity —
worth its own rung later.

Headline research question (superseding "how many states does HH
need"): what information must a reduced dynamical representation
preserve for particular biological behaviors to survive, and how
do state dimension, coordinates, and inductive bias trade off?

## A-seeds final (2026-09-02): boundary dissolves into a
## reliability difference

Five seeds each, identical config (20 ep):
k=3: F1 {0.00, 0.00, 0.34, 0.00, 0.00} — success 1/5, median 0.00
k=4: F1 {0.47, 0.00, 0.64, 0.60, 0.16} — success 4/5, median 0.47

Statement of record: under this GRU training regime both k=3 and
k=4 can reach spiking solutions; k=4 reaches them far more
reliably (4/5 vs 1/5) and with higher median F1 (0.47 vs 0.00).
The original single-seed 3->4 "transition" was NOT a hard state-
dimension boundary; it survives only as a distributional
trainability effect. No significance claims at n=5 — the raw
distributions are the result.

## A0c result (2026-09-02): precision-behavior mapping goes
## MULTIDIMENSIONAL

Sub-1% one-step error (V 0.87% / m 0.64% / h 0.36% / n 0.32%),
4x better than A0b. Behaviors DIVERGED: f-I RMSE 10.1 -> 3.0 Hz
(near-perfect), rebound stays exactly 1, but OU-drive spike F1
0.864 -> 0.738 and V-RMSE 11.4 -> 15.0. Gate NOT passed.

The naive hinge branch ("fidelity still improving -> smooth
curve") is refuted in its scalar form: past ~1% error, aggregate
one-step precision stops predicting all behaviors monotonically.
Two readings, both recorded pending discrimination:
(a) behavior-specific fidelity — fluctuation-driven timing
    depends on subthreshold phase accuracy that spike-weighted
    training trades away; f-I/rebound depend on attractor
    geometry that it buys;
(b) run-to-run variance — A0b/A0c are single runs, and A-seeds
    just quantified what single runs are worth.
Declared next (before running): seeds n=3 on the A0b config and
A0c config; if the F1 ordering is stable across seeds, (a) wins
and "which aspects of the field does each behavior need" becomes
its own rung — the behavior-ordering observation upgraded from
curiosity to mechanism.

## v4 interim framing (2026-09-02, per review): budget compresses
## the state-count gap

Observable track at 60 epochs: k=2 0.365, k=3 0.628 — a 3-state
voltage-only surrogate nearly matches the previous 8-state best
(0.67 at 30 ep). Headline: longer training compresses the
apparent state-count gap; earlier "capacity" differences were
substantially trainability/convergence-speed differences.

Outcome branches for k=4/k=8, declared before they land:
(i) ~0.63-0.68 -> ladder flattens; state beyond ~3 buys little
    at this budget;
(ii) ~0.8+ -> state still matters, but only visible at
    sufficient optimization budget;
(iii) noisy/non-monotone -> observable-track GRUs remain too
    optimizer-sensitive for dimensional conclusions.

Also logged: observable k=2 (0.365) now sits in the same
territory as the CMA-fitted classical 2-state arms (Izhikevich
0.335, AdEx 0.368) — same nominal state count, different
parameterization, supervision, and trainability, converging on
similar timing performance. The eventual 2-state cross-
comparison (learned-observable vs fitted-classical vs derived-
reduction vs latent-field) is becoming the project's cleanest
table.

## A0-seeds result (2026-09-02): the divergence was a lucky seed —
## and the scalar metric decouples from rollout

A0b config, seeds 1-3 (all ~1.5% one-step rel. error, same as
seed 0): F1 {0.736, 0.737, 0.770}, f-I {9.3, 13.6, 9.5}, rebound
{1, 1, 1}. The original A0b (seed 0, F1 0.864) was the OUTLIER;
A0c's 0.738 sits inside the seed band. Reading (b) wins: no
behavior regressed with precision.

Revisions this forces:
1. Hierarchy corrected: at >=1.5% field precision, firing AND
   rebound are robust (5/5 runs); f-I improves with precision
   (~10 Hz -> 3.0 at sub-1%); OU-drive spike timing PLATEAUS at
   ~0.74 (occasional lucky 0.86). Fluctuation-driven timing, not
   rebound, is the binding behavior.
2. Near-identical scalar one-step error across seeds yields
   different rollout F1 (0.736 vs 0.864) — rollout fidelity
   depends on the residual error's location in state space, not
   its magnitude. The "precision" axis needs a geometry-aware
   metric (future diagnostic: error conditional on phase /
   distance-to-threshold; F1 vs time-within-sequence to separate
   phase drift from event failure).
3. Gate still unpassed; the F1~0.74 plateau under OU drive with
   +-2 ms tolerance over 1000 ms rollouts may be partly
   accumulated phase drift — the F1-vs-time diagnostic
   discriminates drift (F1 decays with time) from event failure
   (uniform). Declared as the next A0 analysis.

## GEOMETRIC TRAINING declared (2026-09-02, collaborator proposal):
## replace temporal unrolling with the geometry that generates it

Formulation: z = E(x); latent field G(z, I); decoder D. Train on
INDEPENDENT state samples with three geometric losses, no rollout,
no BPTT:
  recon        ||D(E(x)) - x||^2
  push-forward ||G(E(x), I) - J_E(x) F(x, I)||^2   (JVP, autodiff)
  pull-back    ||J_D(z) G(z, I) - F(D(z), I)||^2   (z = E(x))
The diagram (x -> x_dot) -> (z -> z_dot) is asked to commute.
Integration happens only at evaluation. Spike-region samples
weighted 10x (the formulation admits phase-space weighting — the
direct response to the error-geometry finding). Time is treated
as flow through geometry, not a sequence index; absolute t is NOT
added as a variable (OU forcing is not a function of clock time).

Arms: k in {1,2,3,4,8} x 2 seeds, 20 epochs, width 64 E/D +
128 G. Rollout-trained latent field (the original hh_latentfield
plan) becomes arm A of the A/B when compute frees; geometric arm
B runs first.

Predictions (before running):
P-geo1: k=4 lands in the uncompressed A0 band (F1 0.70-0.80,
        rebound robust) — the diagram-commutes positive control.
P-geo2: k=8 ~ k=4 — no benefit beyond the teacher's dimension
        when full state is observable.
P-geo3: k=2 vs the derived reduction's sanity range (F1 0.325 /
        f-I 26.4 / rebound 1): matching or beating it on
        rebound+f-I means learning FOUND good coordinates;
        timing-good-signature-dead means it reproduces the
        fitted-model pathology.
P-geo4: wall-clock per model far below any rollout-trained arm
        (minibatch regression vs sequential unrolling).
P-geo5: k=1 fails on all signatures.

## Geometric baseline verdict forming + TUBE EXPERIMENT
## preregistered (2026-09-02)

Geo k=4 across seeds: F1 {0.199, 0.031} — bad AND variable, with
spurious rebounds. Diagnosis (collaborator, adopted): tangent
correctness ON the data manifold is not sufficient — integration
error pushes z off the encoded manifold into regions where G has
no supervision; a self-reinforcing excursion follows. Local
tangent knowledge does not determine a robust global flow for an
imperfect learned field: the field must be right in the
neighborhood the imperfect numerical system actually traverses.
P-geo4 (speed) SURVIVED: ~220 s vs ~50 min rollout-free A0 (~14x).

Five arms preregistered, all k=4, seeds {0,1}, same architecture,
train_seconds first-class:
  geo           exact manifold tangents (baseline; already run)
  geo_noise     random latent tube z' = z + eps, targets DECODE-
                GROUNDED: x' = D(z'), target = J_E(x') F(x', I)
  geo_restore   tube + manifold attraction: target -= lambda
                (z' - E(D(z'))), lambda = 1/ms
  geo_onpolicy  supervise states the model ACTUALLY visits in
                short rollouts each epoch (drift-targeted; the
                dangerous directions are anisotropic — train the
                tube where the learner falls off)
  rollout       latent TBPTT reference (accuracy/time anchor)
Question: how little sequential supervision turns a massively
parallel local description of dynamics into a stable global flow?

Related work filed: Lim & Kasim 2022 (ICML AI4Science) — physical
inductive biases as dynamics-constraint LOSS TERMS on neural ODEs
(Hamiltonian/dissipative), no architecture change; a third
mechanism for constraining the off-manifold field (a dissipativity
constraint on G is a natural future arm). Also noted per review:
latent-NODE geometry work warns that local smoothness regularizers
do not automatically improve long-horizon dynamics — encoder/
decoder geometry and latent dynamics must agree.

## Two additions (2026-09-02, user): flow compilation + the
## sufficiency diagnostic

1. COMPILE THE FLOW: geometric training does not sacrifice
   sequential use — z_{t+dt} = Phi_dt(z_t, I) is just the flow of
   the learned field; after geometric training, distill Phi_dt
   into a cheap discrete transition cell T_dt (supervised pairs
   generated by integrating G). Train parallel, run sequential.
   Queued behind the tube verdict.
2. VELOCITY-COLLISION DIAGNOSTIC (declared, implemented,
   running): a compressed latent is a SUFFICIENT STATE only if
   the future is single-valued from (z, I). Measure it directly:
   for trained E, find latent near-pairs at matched input and
   compute the dispersion of their required latent velocities
   J_E(x) F(x, I). High dispersion = no deterministic G of ANY
   quality can serve this encoder — separating "G badly trained"
   from "no valid G exists at this k". This is a lower-bound
   instrument: it bounds the k-ladder from below, independent of
   optimization. Prediction: k=4 low dispersion (an injective
   embedding exists); dispersion rises as k drops; k=1 severe.

## Geometric k=8 reading + tube prediction sharpened (2026-09-02)

k=8 geo: s0 F1 0.049 / V-RMSE 81; s1 0.093 / V-RMSE 328 — yet
s1's f-I RMSE is 18.9: catastrophic amplitude, roughly plausible
oscillation RATES. Reading (per review, adopted): the field
preserves coarse rotational/timing structure while losing the
attracting geometry — approximately the right cycle frequency,
badly wrong orbit radius. Pure tangent training grows unstable
with latent dimension: more unsupervised NORMAL directions to
drift into, even as some coarse invariants survive.

Sharpened tube prediction (P-tube, declared before running):
if the failure is normal-direction drift, tube/restoring training
should fix k=8 MORE than k=4. Tube arms therefore extended to
k in {4, 8} (was k=4 only). Anticipated two-sided landscape:
large k unstable (unconstrained directions), small k stable-ish
but insufficient (distinct futures collapse — measured
independently by the collision diagnostic), a workable middle.
The two instruments (tube arms for the drift side, collision
ratio for the collapse side) now bracket the ladder from both
ends.

## Hypothesis update + memory-as-geometry program (2026-09-02)

UPDATED (k=2 blow-up forces it): off-manifold drift is a GENERIC
failure of tangent-only training across latent dimensions — k
modulates probability/severity but is not the root cause. The
learned on-manifold field lands in different off-manifold
geometries across seeds even when the supervised tangent fit is
acceptable. Raw geometric F1-by-k will NOT be interpreted as a
ladder; the baseline's job (speed win + missing stability
constraint exposed) is done.

Tube outcome branches (declared): (1) rescues every k ->
off-manifold support was dominant; (2) rescues high k only ->
low-k information loss re-emerges once stability is fixed;
(3) fails broadly -> tangent supervision misses something deeper.
Tube arms scored on TWO axes: mean AND cross-seed variance
(consistent 0.65 beats sometimes-0.8-sometimes-explodes).
Collision diagnostic on the full geo ladder separates
representation failure / field-support failure / optimizer
variance; k=2 passing collision despite terrible rollout would
localize failure downstream of the encoder.

MEMORY AS GEOMETRY (user proposal + collaborator refinement,
adopted as the B-track program): convert history into current
geometric state rather than sequential recurrence. Progression
(strictly one ingredient at a time): (D0) delay-coordinate
sufficiency WITHOUT TRAINING — the collision diagnostic applied
to delay vectors (V_t, V_{t-tau}, ...): dispersion of the hidden
(m,h,n) among delay-space near-pairs at matched input, vs random
pairs. Takens predicts dispersion falls as delays are added.
(B2/B3) delay-augmented latent fields if D0 passes; then fixed
reservoir (fading history as state), multi-timescale traces (the
program's oldest primitive, again), Hopfield/prototype memory
only if a task needs discrete recall; associatively-composable
flow maps (parallel-scan structure, S4/Mamba lineage) as the
parallel-training-compatible recurrence. Unifying criterion:
memory is extra geometry whose coordinates make the future
single-valued.

## COLLISION LADDER (2026-09-02): the sufficiency knee is at k=3
## — the cleanest result of the project

Full geo ladder, both seeds (ratio = near-pair velocity
dispersion / random-pair, matched input):
k=1: 0.263/0.270 (p95 7.2/7.4 — catastrophic collisions)
k=2: 0.238/0.237 (p95 1.66/1.62)
k=3: 0.135/0.137 (p95 0.72/0.73)  <- knee
k=4: 0.140/0.139 (p95 0.71/0.70)
k=8: 0.141/0.133 (p95 0.73/0.73)

Monotone, halving between k=2 and k=3, flat floor thereafter;
seed agreement to the third decimal — this instrument involves NO
rollout and no optimization luck in the measurement itself.
Physical reading (restrained): consistent with m being fast-
slaved to V (m ~ m_inf(V)), i.e. HH's EFFECTIVE dimension under
this drive is ~3 — the classical (V, n, h) reduction. k=2's
1.7x-above-floor ratio explains why hand-picked 2-D coordinates
(our reduction control) work while learned 2-D encoders strain.
Caveats: conditioned on this encoder family/objective (E trained
jointly with G); the floor ~0.14 includes legitimate within-
radius variation. Cross-check available: rerun with an E trained
on reconstruction only.

Note the instrument inversion: rollout F1 (optimizer-dominated)
produced no readable ladder; the training-free geometric
diagnostic produced a textbook one. Measurement quality came from
REMOVING learning from the measurement.

## D0 result (2026-09-02): history-as-geometry works, measured

Delay-coordinate sufficiency (no training; dispersion of hidden
(m,h,n) among delay-space near-pairs at matched input / random
baseline): V_t alone 0.347; 2x1ms 0.260; 5x1ms 0.197; 2x3ms
0.211; 3x3ms 0.181; 5x3ms 0.140. Monotone in delay count; longer
lags dominate (HH's slow variables live at 5-20 ms, so 1 ms-
spaced samples are near-redundant).

Headline: FIVE voltage samples spanning 12 ms reach the same
sufficiency floor (0.140) as the k>=3 full-state learned encoders
(0.135-0.141) under the same protocol. Voltage-only insufficiency
(0.347) is measured as the root cause of the observable track's
difficulty; a 12 ms delay window repairs it without recurrence.
Takens vindicated empirically on our exact data. Next rung (B2):
feed delay vectors into the latent-field machinery — state
inference by geometry, dynamics by tangent training, stability by
the tube-arm winner.

## Fixed-representation sufficiency + tightenings (2026-09-02)

Zero-learning arms (same protocol): raw4 0.100 | (V,h,n) 0.086 |
(V,h) 0.092 | (V,n) 0.144 | V alone 0.350 (reproduces D0's 0.347
— cross-instrument sanity holds).

Readings:
1. Hand-picked 2-D coordinates beat the learned k=2 encoder
   decisively on sufficiency (0.09-0.14 vs 0.238) — the
   coordinates result re-derived by an instrument with zero
   optimization at ANY stage.
2. Learned k>=3 encoders (0.133-0.141) sit above the fixed-
   coordinate floor (0.086-0.100): encoder mixing costs
   sufficiency even at adequate dimension.
3. SURPRISE, held with restraint: (V,h) 0.092 vs textbook (V,n)
   0.144 — under this drive, retaining h pins the remaining
   hidden state better than retaining n. Candidate micro-finding:
   the better 2-D reduction may keep (V,h). Needs a dynamics
   check (simulate a (V,h) reduction with n slaved) before any
   claim.
4. The 12 ms delay window (0.140) ties the textbook 2-D
   reduction: history is a substitute coordinate system, measured.

Epistemic tightenings adopted (per review): observable track had
a fundamental information bottleneck BEFORE architecture or
budget entered (not "architecture/budget were irrelevant" — v4
showed they matter too); "explicit learned recurrence is not
necessary for STATE INFERENCE in this HH regime" (not
"recurrence isn't needed" generally); the collision instrument
"eliminates the optimizer from the MEASUREMENT stage" (learned-
encoder arms still measure optimized artifacts; the fixed arms
now provide the zero-learning anchor). k=3 knee stated as "a
three-dimensional sufficiency floor under this dataset/drive/
criterion, consistent with the classical fast-m reduction" — not
"HH's dimension measured".

B2 declared (the first architecture where every ingredient has a
measured reason): delay vector (5 x 3 ms) -> E -> z, dz/dt =
G(z, I), stabilization = tube-arm winner, then flow compilation.
Control: same-dimension delay vectors in short/badly-chosen
windows vs 12 ms — is the WINDOW special (timescale needed to
reconstruct h/n) or only the embedding dimension? Each component
maps to a measured failure: delays fix state ambiguity (0.35 ->
0.14); tangent training fixes training cost (~14x); tube fixes
drift (pending); compilation fixes deployment cost (pending).

## (V,h) dynamics check (2026-09-02): sufficiency and dynamical
## adequacy DISSOCIATE — an instrument-scope result

Simulated (V,h) reduction (n slaved): F1 0.202 / f-I 61.7 /
rebound 1, vs (V,n) (h slaved): 0.325 / 26.4 / 1. The
representation with BETTER static sufficiency (0.092 vs 0.144)
generates WORSE dynamics. Reading: h carries more information
about the remaining hidden state, but n does more dynamical work
(n^4 drives repolarization); evolving the working variable and
slaving the informative one wins.

Scope calibration, stated plainly: the collision diagnostic
measures WHICH COORDINATES IDENTIFY THE STATE (static
sufficiency), not which coordinates' evolution equations
reproduce the flow (dynamical adequacy). Its first out-of-domain
prediction failed, and the failure defines the boundary. Both
axes are needed: sufficiency bounds what any dynamics could
recover; adequacy is a property of the dynamics built on top.
The candidate micro-finding "the better 2-D reduction keeps
(V,h)" is WITHDRAWN for dynamics and retained only as a statement
about static state identification.

## v4 final (2026-09-02): branch (ii) — the observable ladder was
## real, budget revealed it

60-epoch voltage-only GRU ladder: k=2 0.365 / k=3 0.628 / k=4
0.697 / k=8 0.869 (V-RMSE 13.1, f-I 14.5, rebound EXACTLY 1 —
first observable-track model to fire the correct rebound).
Declared branch (ii) confirmed: state still matters; the earlier
flattening was budget-starvation. Notable: voltage-only k=8 TIES
the full-state A0b lucky-seed best (0.864) — consistent with D0:
recurrence acts as an implicit delay embedding, internally
reconstructing the hidden state that a 12 ms window provides
explicitly. Standing caveat: single seed per arm (A-seeds
quantified what that is worth); the k=8 number needs seeds before
strong claims. The B-track historical baseline is now complete.

## Moving-target confound + FROZEN-CHART variant declared
## (2026-09-02, before running)

v4 wording tightened per review: "at fixed long training budget,
larger recurrent state produced a monotone performance trend ON
THIS SEED" — voltage-only observation is not fundamentally
insufficient IF recurrence has capacity+budget to reconstruct
state from history internally (the implicit form of what D0's
delay window does explicitly; the open comparison is
computational: implicit recurrent reconstruction vs explicit
delay geometry, and possibly explicit-for-training ->
distill-to-recurrent for deployment).

Tube confound identified from the rising training loss: tube
targets are decode-grounded through the CURRENT E/D — the
coordinate system and the target both move under G. "Noise tube
fails" therefore admits two readings: (a) random-neighborhood
supervision is insufficient; (b) supervision is useful but
nonstationary. Running preregistered arms continue unchanged.

FROZEN-CHART variant declared: load the geometric checkpoint,
FREEZE E and D (the collision gate already certified these
coordinates sufficient at k>=3), train ONLY G — tube targets
become stationary functions (z', I) -> zdot_target. Quick
discriminator first: frozen noise + frozen restore, k=4 seed 0.
Rising-loss disappearing under frozen charts implicates
nonstationarity; persisting implicates the supervision itself.

Staged pipeline adopted as the project's method template:
Stage 1 discover coordinates (gate: collision sufficiency) ->
freeze -> Stage 2 learn robust dynamics on-manifold + tube
(gate: rollout adequacy + off-manifold stability) -> Stage 3
integrate/compile (gate: deployment cost). One optimizer per
stage, one gate per stage.

## Frozen-chart discriminator (2026-09-02): split verdict

Frozen E/D (k=4 s0): noise 0.193/76.6, restore 0.166/63.3 —
versus joint: noise 0.115/249, restore 0.000/301. Training loss
now decreases monotonically (0.00122 -> 0.00102); the rising-loss
signature was nonstationarity, confirmed. BUT stationary targets
+ certified coordinates still do not stabilize rollout: the tube
supervision as implemented (radius 0.1 std, uniform weight) is
insufficient even without the confound. Two-stage separation
validated methodologically; latent stabilization unsolved.

Strategic reprioritization (declared): the A0 field rolls out
well because x-space is DATA-NATIVE — trajectories fill the
region rollouts traverse and the true field contracts toward the
attractor; learned charts manufacture off-manifold territory that
data-native coordinates simply do not have. Delay coordinates are
also data-native. B2-delay (field learned directly on the 5x3ms
delay vector + I, no autoencoder) is promoted ABOVE further
latent-tube engineering: it may inherit A0's rollout benignity
for free. Latent tubes remain interesting for the compression
question but are no longer the critical path. On-policy joint
arms and rollout references still complete the preregistered
table.

## B2-delay v1 (2026-09-02): FAILED both seeds — strong data-native
## hypothesis falsified

F1 {0.007, 0.172}, V-RMSE {153, 97}, 26 spurious rebounds on s1;
training 117 s (fastest arm yet — the speed result keeps holding).
Diagnosis: in rollout the lag components are the model's own past
predictions — errors feed back through the REPRESENTATION itself
(a mistimed spike corrupts the window for 12 ms), and rollout-
generated windows leave the thin manifold of teacher windows just
as latent trajectories left the encoded manifold. Data-nativeness
of the coordinates was not the distinguishing variable; A0's
distinguishing property is now sharper: it integrates COMPLETE
Markov coordinates whose true field is contracting, so state
error does not corrupt the state representation. Third
independent instance of sufficiency NOT implying rollout adequacy.

B2 v2 declared (before running): denoising supervision — train on
teacher windows with small Gaussian corruption of the lag
components, target = analytic dV/dt at the TRUE underlying state.
Stationary targets, iid, parallel; teaches projection-back-to-
manifold in delay space (the restore idea, without moving
coordinates). If v2 also fails, the honest conclusion is that
one-step/tangent supervision of ANY data-native form is
insufficient for this system's rollout, and sequential
supervision (v4-style, or hybrid mostly-parallel + short-rollout
correction) is genuinely load-bearing.

## B2 v2 verdict + the three-gate hierarchy (2026-09-02)

v2 (isotropic lag denoising, sigma 0.03, targets = clean-state
analytic dV): {0.139, 0.099}, blow-ups softened (74-89 mV vs
97-153) — marginal improvement, no stabilization. Interpretive
note recorded per review: corrupted windows are not valid delay
embeddings of any HH trajectory; v2 deliberately taught a
DENOISING/CONTRACTION rule around the valid-window manifold, not
the literal field at corrupted points. Verdict: the basin an
isotropic rule builds is too small/misshapen — the self-generated
error distribution is structured (anisotropic).

Three increasingly demanding notions, now each with measured
instances: (1) state IDENTIFIABLE from the representation
(collision/D0 gates); (2) local dynamics PREDICTABLE (tangent
losses; cheap, parallel); (3) representation REMAINS VALID UNDER
SELF-GENERATED ROLLOUT (the unsolved gate; v4's GRU solves all
three implicitly at sequential-training cost). B2's purpose
restated: can explicit history geometry pass gate 3 much more
cheaply than implicit recurrence.

B2-onpolicy declared (before running): iterative corrective
training — train on clean tangent samples; roll out briefly from
teacher-primed buffers; collect the ACTUAL malformed windows the
model generates; pair each with the clean teacher derivative at
that time; retrain; grow the rollout horizon per round
(50 -> 100 -> 200 -> 400 steps). Metrics per round: spike F1 AND
time-to-divergence (progressive repair is evidence even before
final metrics look good). Also declared: frozen-chart iterative
on-policy remains the decisive latent-side test (three moving
objects reduced to one), and the hybrid fraction (mostly-iid +
small trajectory supervision) is the quantity to optimize if
gate 3 demands sequential constraints.

## B2-onpolicy result (2026-09-02): progressive repair confirmed,
## rate insufficient

TTD per round: 1.8 -> 2.0 -> 5.2 -> 4.9 -> 8.2 ms; V-RMSE 125 ->
67; F1 ~0 throughout; 104 s total. The iterative corrective loop
REPAIRS the flow monotonically (the diagnostic signature asked
for) but at ~4.5x TTD per 4 rounds while gate 3 needs ~1000 ms —
two orders of magnitude away at this correction budget. First
hybrid-fraction data point: ~100 s of drift-targeted correction
buys 8 ms of stability; ~1 h of sequential training (v4) buys
the full horizon. Gate 3's demand for trajectory-level
supervision now has a measured exchange rate, not just a verdict.
Open per declaration: whether many more rounds/denser collection
change the rate; the frozen-chart latent on-policy remains
undone; rollout references still completing the tube table.

## TUBE TABLE COMPLETE (2026-09-02): final scores and verdicts

Spike F1 per arm (seeds 0,1):
  geo baseline    k4 {0.199, 0.031}  k8 {0.049, 0.093}
  geo_noise       k4 {0.115, 0.263}  k8 {0.000, 0.000}
  geo_restore     k4 {0.000, 0.045}  k8 {0.034, 0.000}
  geo_onpolicy    k4 {0.045, 0.008}  k8 {0.002, 0.045}
  frozen noise    k4 s0 0.193 | frozen restore k4 s0 0.166
  rollout ref     k4 {0.000, 0.000} (6 ep, SUB=2 —
                  BUDGET-STARVED; not a valid upper bound)

Verdicts: joint tube arms land in preregistered branch 3 (fail
broadly) on both axes — means near zero AND variance unchanged.
P-tube FALSIFIED: k=8 tube arms did worse, not better. Frozen
charts remain the only partial rescue. Final architectural
observation: the Euler-integrated latent field (Geo) never fired
under ANY supervision tried (tangent, tube, corrective,
sequential-at-low-budget), while the GRU-cell latent transition
(experiment A) fired at equal state and less budget — the
DISCRETE TRANSITION CLASS of the latent dynamics matters more
than the supervision mode for escaping silence. The GRU's gating
may act as an implicit stabilizer that the raw Euler field lacks.
Fair-fight caveat: the rollout reference needs a full-budget
rerun before "sequential supervision fails for this architecture"
is claimed; queued, not run.

## Session-closing consolidation (2026-09-02, per review)

CONCLUSION TIGHTENED: "trajectory-level supervision is load-
bearing" is NOT yet established. Two blockers: (a) the rollout
reference was budget-starved (F1 0.0 at 6 epochs) — so what it
actually shows is that sequential supervision does not
AUTOMATICALLY solve gate 3 either; architecture, integration and
optimization still matter; (b) B2-onpolicy's monotone repair
(TTD 1.8 -> 8.2 ms) means the mostly-parallel route is slow, not
dead. The demonstrated references for gate 3 remain v4-k8
(0.869), experiment A's GRU arms, and the A0 field.

GATE 3 SPLIT (adopted): 3a DISTRIBUTION VALIDITY (rollout visits
states unlike training) vs 3b ERROR COMPOUNDING / PHASE STABILITY
(every state locally valid, phase drifts — A0's F1~0.74 plateau
with intact firing/f-I/rebound is the 3b signature). Next
B2-onpolicy iteration scores them separately: boundedness, TTD,
PHASE ERROR VS TIME, F1, f-I, rebound. A model on the right
attractor drifting 5 ms out of phase is dynamically far better
than its F1.

EXCHANGE-RATE CURVE formalized as the target quantity:
d(stable horizon)/d(trajectory supervision), normalized by
wall-clock. FAILURE-FRONTIER refinement declared: concentrate
corrective samples at t* (first meaningful divergence) rather
than uniformly over rollouts — grow the basin of autonomous
validity outward from the data manifold, spending correction at
the frontier.

THE PROJECT'S PRODUCT, restated:
  usable dynamical primitive = sufficient representation
    + learnable local flow + stable self-generated evolution.
Instruments exist for the first two; the third is the open
problem, and "local geometry gives ~15x training speedup, plus a
small measured amount of sequential correction for global
stability" — if it holds — is the practically useful result, not
a compromise.

## External reviewer's session verdict (2026-09-02, recorded
## verbatim in substance)

What survives most strongly: (1) representation is measurable
without rollout — instantaneous V ambiguous, short history
repairs it, sufficiency knee ~3 under this regime; (2) local
dynamics: direct field learning ~15x cheaper, real even though
tangent-only rollout is unstable; (3) autonomous evolution is
the unsolved piece — and sequential supervision is NOT yet
established as the clean solution for the field architecture
either; (4) coordinates matter independently of dimension;
(5) history and recurrence are two ways of purchasing state
(v4 implicit, D0/B2 explicit); (6) long-horizon error needs
decomposition (distribution exit vs phase drift).

The next quantitative object is the PARALLEL-SEQUENTIAL EXCHANGE-
RATE CURVE — trajectory information required for stability at 10 /
100 / 1000 ms — not another architecture leaderboard. Favorable
knee -> mostly-parallel + targeted correction -> compile to cheap
recurrent cell. No knee -> temporal composition carries
information local flow matching cannot cheaply substitute — 
equally informative. "Time as geometry" survived contact with
experiment the best way: not as a replacement for time, but as
the decomposition exposing which parts of temporal learning
parallelize and which are tied to self-generated trajectories.

## CREDIT GEOMETRY declared (2026-09-02, user proposal +
## collaborator development): the backward-time dual

Program (C-track, NOT bolted into HH yet per review): a primitive
carrying three state kinds — z (dynamical: what am I doing),
h (history: what past got me here), c (credit/sensitivity: how
would changing me affect future objectives). Lineage: adjoint
states, synthetic gradients (predict future gradients from local
information), e-prop eligibility traces (forward-maintained local
responsibility x later global learning signals — the biological
calcium/CaMKII story). Design law adopted: credit state is
COMPRESSED AND SHARED (a few numbers per neuron/unit, low-rank
sensitivity factors), never per-weight — per-weight credit
recreates the Adam problem the program set out to escape.
Distillation symmetry: expensive ODE integration -> compiled flow
map; expensive BPTT/adjoint -> compiled credit map. The duality,
recorded: history geometry makes the FUTURE single-valued
(forward information problem); credit geometry would make
LEARNING locally available (backward information problem).

## Immediately testable piece: the SENSITIVITY MAP q(x), zero
## learning (declared before running)

For HH the Jacobian of the true RHS is analytic — q(x) is
measurable today. Two parts:
1. Diagnostic: distribution of the local expansion rate
   lambda_max(sym J_F) over the state space — where does small
   error amplify? P-sens1: it concentrates near threshold/
   upstroke, OVERLAPPING but not identical to the hand-guessed
   V > -20 mV spike weight (near-rheobase subthreshold regions
   should get weight the voltage threshold missed).
2. Sensitivity-weighted field training: rerun the A0b config
   replacing the hand-guessed spike weight with w ~ measured
   lambda+. P-sens2 (falsifiable): the OU-timing F1 plateau
   (0.736-0.770 seed band) shifts up. Either outcome informs:
   up -> error PLACEMENT was the plateau's cause and credit
   geometry pays immediately; unchanged -> the plateau is phase
   drift (gate 3b), not weighting, and the sensitivity map's
   value moves to the tube/frontier machinery.

## Sensitivity diagnostic result (2026-09-02): the hand-guessed
## weight aimed PAST the decision

Measured lambda_max(sym J_F) by voltage band (per ms): -90..-70
CONTRACTING (-0.10); rest 0.25; -60..-50 1.17; -50..-20 (threshold
approach) 16.4; -20..+20 (spike body) 24.4; >+20 (peak) 9.0.
Only 45.5% of top-decile sensitivity mass lies inside the
V > -20 mV region every prior run weighted 10x — the majority
sits in the -50..-20 mV DECISION zone, which received no extra
weight in any experiment to date. P-sens1 confirmed with a sharper
reading: what matters is the commitment to spiking, not the spike
event. The credit-geometry premise (measured sensitivity beats
guessed sensitivity) survives its first contact. Part 2
(sensitivity-weighted A0b config, seeds {0,1}, vs the 0.736-0.770
plateau band) running.
