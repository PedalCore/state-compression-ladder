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

## Gate-3 mechanism + the composition recipe (2026-09-02, per
## review; next session's roadmap)

MECHANISM, stated for the record: low one-step error does not
control long-horizon error — the INCREMENTAL STABILITY of the
learned field (does evaluating F-hat at self-generated x+delta
damp or amplify delta) decides rollout fate. Two regimes, mapped
to our results: amplification -> the 100-600 mV blow-ups (3a);
neutral-drift along the orbit -> A0's gentle F1~0.74 plateau with
intact signatures (3b: right attractor, wrong clock — a +-2 ms
criterion eventually scores pure phase drift as failure).
Why tubes failed, sharpened: restore/noise fix NORMAL error;
they do nothing for tangential (speed/phase) error — and
indiscriminate contraction is actively wrong for oscillators
(must preserve neutral motion along the orbit). The principled
version is TRANSVERSE stabilization: penalize expansion normal
to the orbit, leave the tangent direction alone.

RECIPE (adopted as next session's plan):
1. certified representation (12 ms delay / collision gate), frozen;
2. iid/parallel field learning (the 15x win);
3. diagnose local expansion — normal-to-orbit vs tangent
   separately (extends the C-track sensitivity map);
4. short-horizon COMPOSITION curriculum, multiple-shooting style:
   5 -> 10 -> 20 -> 40 -> 80 ms segments with boundary-consistency
   penalties; advance horizon only on success;
5. SEMIGROUP loss: Phi_2dt(x) must equal Phi_dt(Phi_dt(x)) —
   teach that transformations at different horizons compose;
6. failure-frontier sampling (concentrate sequential supervision
   at first divergence, move the frontier outward);
7. metrics split: transverse error AND phase error vs time
   (8 ms phase shift with correct rate/f-I/rebound is NOT the
   same failure as a 593 mV blow-up);
8. INTEGRATOR ISOLATION CHECK (declared, cheap, not yet run —
   field checkpoints must be saved first, noted): same learned
   field under Euler / small-step / RK4 / adaptive reference. If
   stability appears under RK4, part of gate 3 is discretization,
   not learning — critical for compiling the eventual cell.
Expected product: the exchange-rate curve with a hybrid optimum —
lots of cheap local supervision + a little targeted composition
supervision, not a return to hour-long sequential training.

## NCA arm declared (user proposal + collaborator development)

The import is the TRAINING PHILOSOPHY, not the grid: Growing-NCA
hit gate 3 in another field (local rules good over training
horizon, decay/explode when iterated) and solved it with
persistence pools — train from the model's own generated,
corrupted, and damaged states so the target becomes attractor-
like. "Teach recovery, not just correctness."
Experiment (A/B/C, declared): on the certified delay
representation, same budget/params/seeds —
  A: current B2 MLP (concatenated delays);
  B: NCA-delay — each delay sample a cell with 4-8 hidden
     channels, one SHARED local rule (rich local state without
     parameter blowup — the biology bargain in miniature);
  C: B + persistence pool (valid + model-generated + corrupted +
     failure-frontier states).
Score: TTD, F1, f-I, rebound, phase drift, wall-clock, seed
variance. C >> B -> self-repair training matters; B >> A ->
local recurrent organization matters; neither beats B2 -> discard.
Caveat recorded: NCA training is sequential (BPTT through
updates) — it buys inductive bias for self-correction, not
parallelism; scaling vision (population-level NCA, computation as
self-organizing dynamical material) parked for later.

## C-track corrections (2026-09-02, per review) + metric check
## PASSED

1. TERMINOLOGY FIXED: q(x) = lambda_max(sym J) is INSTANTANEOUS
   LOCAL EXPANSION, not phase sensitivity. The phase object is
   the iPRC/adjoint Z(theta) — how a perturbation at a given
   cycle phase shifts future spike timing. Declared as the next
   zero-learning C-track instrument: q_expand (where error
   grows) vs q_phase (where error changes timing) NEED NOT peak
   together, and F1 is a timing metric — q_phase is the one
   aimed at the plateau.
2. METRIC-INVARIANCE CHECK RUN AND PASSED: per-std standardized
   coordinates reproduce the structure (decision zone 13.1/ms,
   spike body 18.2, overlap 0.444 vs 0.451). The -50..-20 mV
   decision-zone finding is not a units artifact.
3. FORK WEAKENED: sensw failure does NOT uniquely identify gate
   3b — it leaves (a) placement not binding, (b) instantaneous
   expansion the wrong sensitivity quantity, (c) phase drift.
   Discriminating between (b) and (c) is exactly what the iPRC
   map is for.
4. Declared arms for the weighting comparison (fair total loss
   mass): plain | old V>-20 | scalar lambda+ | DIRECTIONAL
   (penalize v_max-aligned field error — errors along locally
   expanding directions matter disproportionately) | eventually
   PRC-weighted.
5. The three-map hierarchy adopted as the C-track's shape:
   expansion map (trajectory blow-up) -> phase map (timing) ->
   task/credit map (eventual loss). Current q is a TEACHER
   ORACLE, per the house pattern: exact expensive object ->
   measure whether it helps -> distill into a cheap carried
   predictor q-hat(z) only if it does. The grounded primitive:
   z (dynamics) + h (history) + q (sensitivity/credit).
Standing result regardless of sensw outcome: the hand-designed
weight emphasized the EVENT; measured geometry says much of the
expansion lives in the pre-spike DECISION region — plausible
intuition replaced by measurement, which is the project's job.

## P-sens2 scored (2026-09-02): NULL — no plateau shift

sensw-field: {0.735, 0.788} vs A0b-config band {0.736, 0.737,
0.770} (+0.864 lucky seed). Seed 0 at band center, seed 1
marginally above band max — within seed variance at these n.
Rebound correct both seeds, f-I 9.3/14.0. Verdict: scalar
lambda+-weighted training does NOT move the OU-timing plateau,
even with measured (vs guessed) placement. Under the weakened
fork this does not uniquely identify the cause; the declared
discriminators stand: (b)-vs-(c) is the iPRC map's question
(is timing error concentrated where q_phase, not q_expand,
peaks?) and the directional-weighting arm tests whether the
scalar-vs-vector distinction was the miss. The standing
diagnostic result (decision-zone geometry, metric-robust) is
unaffected by this null.

## NCA arm predictions (2026-09-03, declared before running)

Design pinned: 5 cells (one per certified delay position) x 8
channels; V-channel of the newest cell is the prediction; ONE
shared local rule f(s_left, s_i, s_right, I) applied each 0.1 ms
step; history maintenance is the rule's job (a learned delay
line — recurrence inside geometry, literally). Arms B (NCA,
TBPTT) and C (B + persistence pool: 30% of chunks start from
model-generated pool states, 20% of pool starts get a damaged
cell) x seeds {0,1}, 40 epochs. Arm A baseline = B2 v1 (already
run). Honest caveats recorded: this is SEQUENTIAL training (the
NCA import is inductive bias, not parallelism), and the shared
rule (~2.2k params) exceeds v4-k8's 930 — the comparison is
about state ORGANIZATION, not parameter count.

P-nca1: B reaches the sequential-training class (F1 > 0.6) —
        shared-local-rule state competitive with monolithic
        recurrence at equal budget.
P-nca2 (the point of the arm): C >= B on time-to-divergence AND
        lower cross-seed variance — persistence/damage training
        buys gate-3 robustness that correctness training alone
        does not.
P-nca3: C's advantage, if any, shows most in free-run stability
        metrics (TTD, boundedness), not in teacher-adjacent F1.

## NCA interim (2026-09-03): arm B fails; arm C had a DEAD POOL
## (bug, caught, fixed)

Arm B (pure correctness): {0.09, 0.181}, TTD < 1 ms, blow-ups —
P-nca1 fails at 40 epochs (budget caveat standing). Arm C's first
run came back BYTE-IDENTICAL to arm B seed 0 — the persistence-
pool restart condition could never fire (pool entries store later
chunk boundaries; restarts were only attempted at the first).
Dead code, exposed by identical floats. Fixed (restarts at every
chunk boundary); arm C reruns follow. P-nca2 remains unscored —
the b-vs-c comparison hasn't actually run yet.

## NCA arm scored (2026-09-03): all predictions failed — discarded
## at this budget/design

Arm B (correctness only): F1 {0.09, 0.181}, TTD {0.9, 0.5} ms.
Arm C (persistence pool, CORRECTED after the dead-pool bug):
F1 {0.002, 0.000}, TTD {1.3, 0.2} ms, drift magnitude mixed
(110 vs 504 mV). P-nca1 FAILED (nowhere near the sequential-
training class); P-nca2 FAILED (no consistent TTD gain, no
variance reduction — seed 1 worse on every metric); P-nca3 moot.
Per the declared protocol (neither beats B2 -> discard): the
5-cell delay-geometry NCA is discarded at 40 epochs. Standing
caveats, honestly: budget (40 ep vs the 60 that mattered for v4),
one design point (5x8 cells, one damage scheme, one pool policy)
— the persistence-training PHILOSOPHY is not refuted, but this
instantiation earned no follow-up priority. The gate-3 critical
path remains the composition recipe (multiple shooting, semigroup
loss, failure frontier, transverse/phase split).

## NCA closing reading (2026-09-03, per review) — the pattern
## behind the negatives

The narrow, correct statement of the result: at this design point
and budget, a local shared-rule architecture + persistence-pool
recovery training did not solve gate 3. The diagnostically
interesting piece: arm C seed 0 cut drift magnitude (270 -> 110
mV) while useful dynamics vanished — the persistence objective
taught CONTAINMENT without teaching the right flow, echoing the
tube arms. Cross-M13 pattern, elevated to a working law:

    staying nearby != evolving correctly.

Recovery-style objectives (noise tubes, restoration, persistence
pools) attack where the trajectory is ALLOWED TO LIVE; the
remaining problem is how it COMPOSES through time while
preserving orbit and clock. Decision affirmed: no NCA
architecture search (bigger grids/channels/damage schedules
would turn a clean negative into a fishing trip). The sharpened
gate-3 statement: the model does not primarily need to learn how
to return after damage — it needs the locally accurate rule
CONSTRAINED so that repeated application preserves the correct
orbit and clock. Critical path: parallel field fit ->
short-horizon composition (multiple shooting + semigroup) ->
failure-frontier correction -> longer horizon, with phase/
transverse metrics separated. The ledger is positive: plausible
explanations killed, search space smaller, next experiments
attack temporal composition itself rather than proxies for it.

## COMPOSITION EXPERIMENT declared (2026-09-03, before running)

Design (one mechanism per arm; semigroup loss reserved as arm 2):
Stage 0: iid field training, A0b config (deriv, width 256,
  spike-weighted, 40 ep), checkpoint SAVED (fixes the no-saved-
  fields gap).
Stage 0.5: INTEGRATOR ISOLATION on the stage-0 field — identical
  weights rolled out under Euler (10 substeps, current) vs RK4
  (0.1 ms step): if stability/F1 changes materially, part of
  gate 3 was discretization.
Stages 1-4: SHORT-HORIZON COMPOSITION curriculum — segments of
  H = 5, 10, 20, 40 ms starting from TEACHER states (multiple-
  shooting style; parallel across segments, BPTT only inside a
  segment; Euler 5 substeps in training, standard 10 at eval),
  full-state trajectory loss, 3 epochs per stage, lr 3e-4 -> 1e-4.
Metrics after each stage: F1, V-RMSE, TTD, f-I, rebound, and
  F1-BY-TIME-WINDOW (4 quartiles of the 1 s test rollout — the
  declining-vs-flat discriminator for phase drift vs event
  failure). Seeds {0, 1}.

Predictions:
P-comp1: integrator check comes back NULL — RK4 ~ Euler-10 (the
  field's ~1.5% error dominates discretization error at 0.01 ms).
P-comp2 (the point): composition fine-tuning lifts F1 above the
  0.736-0.770 plateau band.
P-comp3: stage-0 F1-by-window DECLINES across the second
  (confirming 3b drift); composition training flattens the
  decline.
P-comp4: TTD grows with stage horizon — the frontier moves out.

## Composition v1 (2026-09-03): stage 0 delivered two verdicts,
## stage 1 exposed a design bug — killed and fixed

Stage-0 verdicts (stand regardless): (1) P-comp1 CONFIRMED NULL —
RK4-0.1ms vs Euler-10 on identical weights: F1 0.727 vs 0.742,
same TTD/f-I/rebound. Discretization is not a gate-3 component;
the compiled cell won't inherit Euler artifacts. (2) P-comp3
REFUTED, and with it the 3b-drift reading of the plateau:
F1-by-window is FLAT ({0.757, 0.723, 0.777, 0.706}). The plateau
is a THIRD thing nobody had listed: a stationary per-event miss
rate (~25% of near-threshold decisions) with each error locally
HEALED by the entraining drive (input-driven reliability, the
Mainen-Sejnowski phenomenon, working in our favor). Errors recur;
they do not compound. Also: best f-I ever (1.4 Hz).

Stage 1 (5 ms segments) DESTROYED the model (0.742 -> 0.028):
catastrophic forgetting by DESIGN BUG — the segment loss REPLACED
the tangent objective instead of joining it, unlearning the field
everywhere segments don't constrain. (Stage 2's partial rebound
to 0.301 was the curriculum re-learning from wreckage — not the
designed experiment.) Run killed by PID inspection. Fix declared
before rerun: JOINT loss (segment trajectory term + iid tangent
anchor term each update — literally the mostly-parallel + some-
trajectory hybrid formula), segment lr 3e-4 -> 1e-4, stage-0
checkpoint reused. P-comp2/4 remain unscored pending v2.

The stationary-miss-rate finding also pre-registers a revised
expectation: if errors do not accumulate, long-horizon
composition training may buy little — the discriminating question
v2 actually answers.

## Gate 3 RESTATED (2026-09-03, per review): event-decision
## fidelity, not rollout stability

New split: 3a EVENT FIDELITY (does the learned field cross the
same qualitative decision boundaries as the teacher?) vs 3b'
COMPOSITION STABILITY (do errors accumulate once decisions are
correct?). Evidence points to 3a limiting and 3b' largely SOLVED
BY THE INPUT for this driven regime: entrainment washes errors
out. TTD is downgraded as a proxy (it detects divergences the
system later repairs). Composition-v2's degradation is now
EXPECTED: multiple shooting corrects a non-accumulating problem
by perturbing an already-good global field. If seed 1 confirms,
the multiple-shooting branch STOPS.

Declared next (all zero-training, on existing artifacts):
E1. MISS AUTOCORRELATION: P(miss n+1 | miss n) vs P(miss) — if
    ~equal, misses are independent stationary events and the
    locally-healed story is confirmed. Plus T_recover: time from
    mismatch until model-teacher re-synchronization (measures
    entrainment directly).
E2. DECISION-CONDITIONED COLLISION: the sufficiency instrument
    restricted to the decision band (-50..-20 mV, matched I) with
    a BINARY label (teacher spikes within horizon H): do delay-
    space near-neighbours agree on the outcome? Event sufficiency
    vs global sufficiency — if ambiguity concentrates at the
    boundary, the 12 ms representation lacks RESOLUTION for
    decisions (fix: window/lag/extra observable), not state.
E3. PRE-EVENT ERROR ANATOMY: classify events (hit / miss / FP /
    shifted); compare field error magnitude AND direction, hidden
    state, and delay-ambiguity in 5-15 ms pre-event windows. If
    field RMSE is IDENTICAL between correct and wrong decisions,
    scalar field accuracy has hit the wrong target, and the
    useful credit map is the DECISION MARGIN m(x) = distance to
    the spike/no-spike boundary — not lambda_max.
Also declared: gradient norm-ratio + cosine logging between
segment and tangent terms before any further hybrid tuning.
Boxed: Gate 3 = can locally accurate dynamics preserve discrete
event decisions? — with entrainment preventing event errors from
becoming instability.

## E1-E3 results (2026-09-03): the plateau is EPISODIC — and both
## representation and scalar field error are exonerated

E1: P(miss) 0.294, P(miss | prev miss) 0.692 — misses strongly
autocorrelated. The independence half of the locally-healed story
is REFUTED: the failure unit is a DESYNCHRONIZED EPISODE (2-3+
spikes), entered rarely, exited by entrainment (T_recover median
5.4 ms, p90 32 ms). Flat F1-by-window survives (episodes evenly
distributed, non-accumulating).
E2: decision-band delay near-neighbours agree on the spike
outcome 99.2% (disagreement 0.008 vs 0.496 random; ratio 0.015).
Event sufficiency EXCELLENT — the 12 ms window resolves decisions;
representation exonerated.
E3: pre-event whitened field error median 0.0174 (hits) vs 0.0176
(misses) — IDENTICAL. Scalar field accuracy does not distinguish
correct from incorrect decisions. Confirmed: the wrong target.

Corrected causal chain: rare initial decision error (cause not in
scalar error) -> state desync -> consecutive misses while
desynced -> re-entrainment -> hits resume. F1 ~ 1 - (episode
rate x episode length). Two levers, both measurable:
L1 EPISODE ENTRY — E3-v2 declared: analyze the FIRST miss of each
   episode specifically; compare field-error DIRECTION (decision-
   normal component) and margin against matched hits. Scalar
   failed; the direction/margin hypothesis is what remains.
L2 EPISODE DURATION — is 5.4 ms the drive's entrainment constant
   (fixed) or shortenable? Compare teacher-vs-teacher-perturbed
   re-entrainment as the physical bound.

## Composition v2 final + E3-v2: THE PLATEAU MECHANISM FOUND
## (2026-09-03)

Composition v2 final: seed 0 monotone harm (0.742 -> 0.293);
seed 1 TRANSIENT BREAKTHROUGH — 5 ms stage lifted 0.732 -> 0.802
(best OU-timing F1 of the project; checkpoint comp_stage1_s1.pt
SAVED) then collapsed (0.363) and partially recovered (0.667).
Verdict: composition fine-tuning is violently unstable — a small
dose CAN exceed the plateau (so the plateau is NOT irreducible at
this field size) but the procedure is unreliable. P-comp4 failed
(no TTD growth). Branch closed per criterion; gradient-conflict
logging remains the post-mortem tool if ever reopened.

E3-v2 (episode-entry anatomy): the entry misses have a
DIRECTIONAL gate-error signature. Separations (|dmedian|/IQR,
hit vs entry): e_m 1.16 (SIGN FLIP: +0.0017 hits vs -0.0047
entries — the model systematically UNDERESTIMATES dm/dt before
missed spikes), e_n 1.00, e_h 0.95, e_V 0.48; scalar norm
identical (E3). Entries sit at lower h (low sodium availability
— marginal decisions). Mechanism: I_Na ~ m^3 h; a lagging m
during upstroke initiation at a low-margin state kills the spike.
Same-size error, wrong DIRECTION, at marginal decisions — the
~0.74 plateau explained.

Prescribed next (declared): directional decision-band training —
weight the m/h/n COMPONENT errors (signed-aware) in the measured
decision band; predicted to beat scalar lambda+-weighting (which
failed: P-sens2 null) because it targets the discovered
signature. The decision-margin credit map q(x) = margin-to-
boundary gets its concrete first job.

## DIRECTIONAL FIX declared (2026-09-03, before running)

Training: from-scratch A0b protocol (deriv field, width 256,
40 ep, standard spike-region weight) + COMPONENT weighting in the
measured decision band (V in -50..-20 mV): m-error x10, h/n-error
x5, V-error x1. Symmetric loss (no asymmetric anti-bias term —
risk of inducing the opposite bias); the component boost targets
the discovered signature (dm/dt underestimation at low-margin
decisions). Seeds {0,1}. Post-hoc: rerun the E3-v2 anatomy on the
trained model — the mechanistic closure test.

P-dir1: F1 exceeds the A0b band and the scalar-weight arms
        (> 0.788) on at least one seed; both seeds above band
        center.
P-dir2: the episode-entry e_m separation/bias SHRINKS vs stage-0
        (1.16 -> substantially less) — if F1 rises AND the
        signature shrinks, the causal loop closes.
P-dir3: f-I and rebound preserved (band-local component weighting
        must not damage global structure — the failure mode that
        killed composition training).

## Directional fix scored (2026-09-03): MARKER, NOT CAUSE

P-dir2 CONFIRMED: the intervention removed its target — em
separation 1.16 -> 0.68/0.32, en 0.99 -> 0.42/0.01 across seeds.
P-dir1 FAILED: F1 {0.757, 0.703} straddles the band; plateau
unmoved. P-dir3 HALF-FAILED: f-I preserved (1.5/3.3), rebound
LOST both seeds (h/n band-weighting distorts recovery dynamics).
Episode structure reshaped, not shrunk (s0: entries 87 -> 108,
continuations 164 -> 125).

The lesson, now twice-taught (P-sens2, P-dir1): weighting
interventions keep REMOVING their measured targets without moving
F1. The E3-v2 signature was downstream of the true cause. Sharper
diagnosis forced: every anatomy instrument so far evaluates the
field ON THE TEACHER TRAJECTORY, but misses happen on the MODEL'S
trajectory — the causal desync likely begins between spikes, in
ways invisible to on-teacher diagnostics. Declared next
instrument: TRAJECTORY-DIVERGENCE ANATOMY — measure model-vs-
teacher state divergence in the inter-spike interval BEFORE each
entry miss (where does the rollout actually leave the teacher's
path, in which state components, how early?) — moving the
microscope from the field to the rollout itself. The plateau's
cause remains unfound; three non-causes are now measured
(representation, scalar error, m-bias marker), which is how the
space shrinks.

## TRAJECTORY-DIVERGENCE ATLAS declared (2026-09-03, before
## running) — instrument, not fix

Question: starting from a synchronized state, what is the FIRST
statistically reliable divergence between model rollout and
teacher that predicts an eventual entry miss? Walk backward from
each event (tau = 2..50 ms), per-component clock-aligned
divergence D_j(tau) = (mu_entry - mu_cc)/pooled dispersion, plus
PHASE-ALIGNED distance (nearest teacher point in +-3 ms — 
distinguishes wrong-state from time-displaced), plus divergence
amplification A(tau). Events split into FOUR transitions:
correct->correct, correct->miss (ENTRY), miss->miss
(CONTINUATION), miss->correct (RECOVERY) — likely different
phenomena. Run on stage-0 AND the dir checkpoints.

Pre-registered sub-hypothesis (collaborator), with immediate
partial support from existing counts: the directional fix
repaired CONTINUATION, not entry — stage-0 cont-fraction ~0.65 /
entry-rate 0.126 vs dir_s0 ~0.54 / 0.148. If confirmed, the
dir-fix headline upgrades from "marker not cause" to "the dm/dt
signature participates in miss PERSISTENCE, not initiation" — 
mechanistic change hidden under a flat scalar score. Candidate
unifying object: state-dependent AMPLIFICATION of ordinary-sized
pre-existing rollout error, which would reconcile every negative
so far. The research pattern: stop intervening on the latest
visible symptom (the system reorganizes around it); walk backward
to the earliest causal bifurcation.

## ATLAS RESULTS (2026-09-03): initiation lives in the inter-spike
## recovery dynamics

1. Persistence-vs-initiation CONFIRMED: dir-fix reduced
   continuation (0.653 -> 0.536, s0) while raising entry rate
   (0.141 -> 0.170/0.218). Flat F1 hid a real trade. Headline
   upgraded: the dm/dt signature participates in PERSISTENCE, not
   initiation.
2. Entry chronology: divergence reliably visible by -5 ms
   (norm 0.79 sigma), maximal phase-aligned at -2 ms (1.26 — 
   wrong-STATE, not time-displaced), with consistent signs:
   dn > 0, dh < 0, dV < 0 — the rollout drifts UNDER-EXCITABLE
   (excess K-activation, depleted h) during the inter-spike
   interval, so marginal spikes fail. Amplification discriminates
   (amp_100 0.78): ordinary errors preceding misses GROW faster —
   state-dependent amplification supported.
3. The locational irony: every loss to date weighted the spike
   and decision band; the causal drift accumulates in the
   SUBTHRESHOLD RECOVERY region (n/h relaxation, -70..-50 mV,
   weighted 1x throughout). The instrument-first strategy paid:
   the fix candidates were all aimed downstream of the cause.

RECOVERY-BAND ARM declared (before running): dir-fix protocol but
band = -70..-50 mV with n/h component errors x10 (m x1) — aimed
at the measured drift, upstream of initiation. P-rec1: entry rate
DROPS below stage-0's 0.141 (the causal test); P-rec2: F1 exits
the band upward if initiation dominates the budget; P-rec3:
rebound preserved (recovery dynamics are what produce it — this
weighting should HELP it). Atlas rerun on the trained model
closes the loop.

## Recovery-band arm scored + the weighting program closes
## (2026-09-03)

rec-field: F1 {0.734, 0.746} (in band), f-I {8.7, 10.0}, rebound
1/1 (P-rec3 CONFIRMED — upstream weighting is benign where
downstream weighting broke the rebound). Entry rate 0.130/0.140
vs 0.141 (P-rec1 marginal — right direction, tiny); continuation
0.593/0.625 vs 0.653. P-rec2 FAILED.

THE WEIGHTING PROGRAM CLOSES with a coherent negative: three
placement interventions (scalar-sensitivity, decision-band
directional, recovery-band upstream) all null-to-marginal on F1 —
and A0c already showed that 3x LOWER total error (sub-1%) stays
in the same band. Neither error size nor error placement moves
the plateau WITHIN this architecture. What has exceeded it:
a lucky seed (0.864), a transient 5 ms composition dose (0.802),
and a different dynamics class (GRU, 0.869, sequential training).
Working conclusion (restrained): the ~0.74 plateau is STRUCTURAL
to the MLP-field + explicit-integrator formulation under iid
training on this drive — plausibly because a recurrent
integrator learns its own state coordinates in which marginal
decisions are less marginal, while the fixed x-space field
inherits the teacher's razor-edge decision geometry along with
its dynamics. The causal atlas stands as the arc's product:
initiation = inter-spike under-excitability drift, amplified
state-dependently; persistence = the dm/dt mechanism (repairable);
recovery = entrainment (~5.4 ms).

Next-session queue (updated): (1) hybrid architecture test —
learned x-space field + tiny learned correction state (the
credit/memory slot, now with a measured job: absorb the marginal-
decision geometry); (2) stable low-dose composition (the 0.802
exists — find the reliable schedule); (3) flow compilation of the
best checkpoint; (4) R3 adapting teacher; (5) round-5 post after
any of these lands.

## HYBRID FIELD + CORRECTION STATE declared (2026-09-03, before
## running)

Architecture: FROZEN stage-0 field F(x, I) + tiny recurrent
corrector — GRUCell([x, I] -> c in R^8) updated once per record
step, additive field term delta = Head(c) (Head ZERO-INITIALIZED:
training step 0 IS the 0.742 baseline; the design cannot lose to
it). Only the corrector (~500 params) trains, sequentially
(TBPTT, full-state loss, spike-weighted) — the hybrid formula as
an architectural split: parallel-trained bulk + sequentially-
trained tiny state. The corrector's measured job: counteract the
inter-spike under-excitability drift and soften the inherited
razor-edge decision geometry. Seeds {0,1} on their respective
stage-0 checkpoints.

P-hyb1: F1 exceeds the band (> 0.77) on both seeds — the first
        STABLE exceedance (composition's 0.802 was transient).
P-hyb2: entry rate drops materially (< 0.12) — the drift is
        counteracted where weighting could not.
P-hyb3: rebound + f-I preserved (frozen field + zero-init
        guarantee the floor).
P-hyb4: corrector training normalizes the exchange rate: minutes
        of sequential supervision on hundreds of params, vs v4's
        ~1 h on the whole model.

## Hybrid refined into the MARGIN-STATE LADDER (2026-09-03, per
## review; declared before running)

Scoped conclusion adopted: "within the tested MLP-field +
explicit-integrator formulation, under these drives and budgets,
the plateau behaves like a coordinate/model-class limitation
rather than a field-error limitation" — stronger than 'training
failed', safer than 'mathematically intrinsic'.

The correction state is NOT a generic latent: it is an
EXCITABILITY-MARGIN state. dx/dt = F_frozen + Head(c) (zero-init),
c updated recurrently from (x, I); c starts 1-D. Ladder arms,
same protocol/params-budget/sequential training throughout:
  A. frozen field alone (0.742 baseline)
  B. + STATIC feedforward correction (no memory) — the capacity
     control: if B ~ C/D, the ingredient was capacity, not state
  C. + 1-state recurrent correction
  D. + 2-state recurrent correction
  (E. the running 8-D GRU arm = capacity-ceiling reference)
All corrections carry L_corr = lambda * |delta|^2 (lambda 0.1) —
rewarded for leaving the field untouched except where the state
earns its keep; c should be ~0 outside marginal regimes, making
the corrector INSPECTABLE.

Falsifiable activation prediction (the strong one): |c| rises
during the inter-spike interval BEFORE episode-entry decisions,
discriminates impending misses from correct events at -5 ms, and
collapses after recovery. Scored alongside the four-way outcome:
performance (stable band exit, not another transient), initiation
(entry rate down), persistence (no entry/continuation trade),
preservation (f-I + rebound intact).

## HISTORY-CONTAMINATION LADDER declared (2026-09-03, before
## running)

Positive results consolidated per review: history-as-information
WORKED (D0: 12 ms window reaches the sufficiency floor; v4's GRU
proves implicit history suffices behaviorally at 0.869). What
failed was the SIMPLE FIELD OVER EXPLICIT HISTORY (B2) under
self-generated buffers. The clean split: the history BUFFER need
not be learned (deterministic shift register); only the newest
value is model-generated, and contamination spreads lag by lag.

Experiment: B2 checkpoints rolled out at contamination levels
g = 0..4 (g = number of lag slots filled from the model's own
history, most-recent first; the rest supplied by the teacher;
the integrated V_t is always the model's). F1 vs g.
P-contam1: F1 falls monotonically with g.
P-contam2 (the decisive fork): if g=0 (oracle lags) F1 is HIGH
(>0.8), the delay field itself is good and self-contamination is
the failure — the formulation is rehabilitated pending better
V-prediction; if g=0 also sits low, the H -> dV map lacks
marginal-decision precision regardless of contamination.

## Hybrid confound + safety note (2026-09-03, per review)

Recorded: seed and lambda changed together in the running pair —
seed 0 vs seed 1 cannot prove the penalty is load-bearing; a
MATCHED same-seed lambda=0 vs 0.1 ablation is declared and queued
(hyb seed 0, lambda 0.1). Seed 0's lambda=0 trajectory already
teaches two things: the corrector CAN touch the intended
mechanism (ep3: entry 0.141->0.133, cont 0.653->0.576, F1 0.751)
and unconstrained it rewrites dynamics the frozen field had right
(ep6+: F1 0.583->0.372, f-I wrecked). Early stopping is NOT the
solution — the OPTIMUM itself must be safe, or the architecture/
objective is underconstrained. Delta-logging declared: ||delta||
by voltage band + by time-to-event; the cartoon success picture
is delta ~ 0 almost everywhere, rising during the ISI drift
before marginal decisions, vanishing after entrainment — the
credit state observed doing its predicted job, or not.

## Contamination ladder result (2026-09-03): the delay-FIELD is
## dead at the map level — contamination exonerated

g=0 (ALL lags oracle-true): F1 0.004 (b2_s0) / 0.006 (denoised) —
the map fails with PERFECT history. P-contam1 falsified (non-
monotone); P-contam2 resolves to the harsh branch: the
(H, I) -> dV map lacks the precision/stability to maintain even
the current V trajectory between lag anchors, independent of
buffer quality. Quirk recorded: the denoised model scores BETTER
with slightly contaminated lags (g=1: 0.26) than pristine ones —
trained-on-corruption makes oracle history off-distribution.

The three-way dissociation this completes:
  delay coordinates are STATICALLY SUFFICIENT (D0: 0.140)
  a GRU over the same information WORKS (v4: 0.869)
  a memoryless field over the same information FAILS UTTERLY
  (0.004 even with oracle context)
Same information three ways: what differs is the COMPUTATIONAL
FORM. Recurrent integration of history succeeds where direct
functional readout of history cannot — the strongest single
piece of evidence in the project that internal state is not
merely an information cache but a COMPUTATIONAL RESOURCE.
The B2 formulation closes; the internal-state route (GRU, hybrid
corrector) is the road.

## Two branches declared (2026-09-03, per review)

The thrice-replicated rise-then-collapse (composition 0.802,
hybrid lam=0 0.751, hybrid lam=0.1 0.838) is reframed: THE GOOD
SOLUTION EXISTS IN PARAMETER SPACE; trajectory-MSE does not have
its optimum there. Architecture question and objective question
separated:
BRANCH 1 (architecture, runs now): val-F1 checkpoint selection —
  standard model selection on the untouched val split, matched
  seeds {0,1} at lam=0.1. Success = reproducible >0.74-ceiling
  test F1 with preservation. Closes the seed/lambda confound too.
BRANCH 2a (trust region, runs second): replace the soft penalty
  with a HARD bound delta = eps(x)*tanh(raw), eps large only in
  the measured drift+decision region (-70..-20 mV), ~zero
  elsewhere — the corrector structurally cannot rewrite rebound
  or f-I. The cartoon operationalized.
BRANCH 2b (declared, next round): event/decision-aware objective
  — short-horizon spike-hazard target p(spike within tau | x,c,I)
  trained around the low-margin band, plus transition loss
  -log P(correct event at next decision): turn the causal anatomy
  INTO the objective, so the good region becomes an optimum
  rather than a waypoint.
Conceptual finding recorded at full strength (per review): the
contamination dissociation shows history is sufficient
INFORMATIONALLY but not COMPUTATIONALLY — a recurrent state
continuously transforms history into a usable coordinate system;
a static map over the same buffer leaves the decision geometry
razor-thin. The hybrid's meaning: accurate physical field + tiny
learned recurrent coordinate correction, not physics replaced by
recurrence.

## THE TRUST-REGION HYBRID PASSES THE ORIGINAL GATE (2026-09-03)
## — both seeds, F1 0.903 / 0.911

Full branch table (frozen stage-0 field + 396-param corrector,
val-F1 checkpoint selection throughout):
  hyb-rec8-sel        s0 0.712 (never rose)   s1 0.859
  hyb-rec8-TRUST-sel  s0 0.903                s1 0.911
Trust arms: entry rate 0.066/0.067 (baseline 0.141), continuation
0.463/0.350 (0.653), f-I 0.6/0.4 Hz (project bests), rebound 1/1,
windows uniformly ~0.9, cross-seed spread 0.008 — the tightest
training result in the project. delta_by_band, both seeds: ~0.001
in hyperpolarized and spike bands (the hard eps cap), active ONLY
in drift (0.008-0.011) and decision (0.012-0.014) bands — the
credit state OBSERVED doing exactly the job the causal anatomy
predicted. And the day-one instrument gate (F1 > 0.9), failed by
every model for the project's entire history, is passed by both
seeds.

The decisive mechanism: the hard bound did not merely protect the
field — it made the good region RELIABLY REACHABLE (seed 0, which
never rose under soft penalty, reached 0.903). Constraining the
search space to field-respecting corrections turned a lucky
transient into a stable, replicated optimum-adjacent capture.

THE COMPLETE RECIPE, every ingredient measured into place:
  1. parallel tangent-trained field (15x cheap): dynamics, rates,
     rebound, f-I;
  2. entrainment (free, from the drive): long-horizon composition;
  3. 396-param recurrent corrector, HARD-BOUNDED to the measured
     drift+decision region, ~30 min sequential training,
     val-selected: event fidelity 0.74 -> 0.91.
Total: gate 3 passed at ~1/2 of v4's sequential cost concentrated
on 0.6% of the parameters, with the physics interpretable and the
correction inspectable.

## CORRECTOR CAPACITY LADDER declared (2026-09-03, before running)

396 params was the scaffold, not the primitive. Scaling note
recorded: the corrector is a SHARED LAW — N neurons cost
396 + N*k (state), not 396*N; the biology bargain. Compression
question now: how little extra machinery captures what the field
misses? Ladder: kc in {4, 2, 1} (= 152 / 66 / 32 params; runtime
state k = kc), trust + val-selection, seeds {0,1}; kc=8 (396)
already done at 0.903/0.911. Scored on F1-per-parameter and
F1-per-state. P-cap1: kc=2 retains most of the gain (the
anatomy's excitability-margin story needs ~1 slow coordinate);
P-cap2: kc=1 still beats the 0.742 baseline materially; P-cap3:
if kc=1 ~ kc=8, the eventual primitive is ~32 shared params +
ONE extra scalar state per neuron — the strongest possible form
of the answer.

## Post-gate consolidation (2026-09-03, per review)

Framing recorded: the result's weight is the MECHANISM, not the
0.91 — the neuron decomposed as F_base (bulk dynamics, parallel-
trained) + bounded deltaF (marginal decisions only), with the
hard bound structurally preventing interference; the learned
correction OBSERVED silent outside the drift/decision region.
The base field was never incapable; it lacked a small amount of
state for marginal excitability decisions. Unconstrained capacity
was dangerous; soft penalty delayed; the trust region made the
optimum safe AND reachable. "First genuinely successful learned
neuron formulation — demonstrated mechanism, not final primitive."

STATIC CONTROL ARM added to the ladder (declared): static
correctors at matched budgets (widths 3/7/24 ~ 34/81/268 params)
under the SAME trust region + selection — discriminates extra
nonlinear capacity from extra DYNAMICAL STATE at the winning
protocol. If tiny static matches tiny recurrent, the state story
weakens; if 1-2-state recurrent beats equal-size static, M13's
original question gets its strongest answer.

GENERALIZATION declared (R-general, later): replace the hand-
measured eps(V) bands with eps(x) = g(decision margin / learned
credit map) — the principle "protect well-modelled dynamics;
allocate recurrent correction only near qualitative decision
boundaries" made regime-independent. Next milestone: 0.90+ with
the fewest recurrent states and shared parameters.

## CROSS-TABLE COMPLETE (2026-09-03): the correction compresses to
## ~34 params — and form barely matters in complete coordinates

Recurrent (trust+sel): kc8 396p {0.903,0.911} | kc4 152p
{0.908,0.912} | kc2 66p {0.897,0.910} | kc1 32p {0.868,0.876}.
Static: w24 244p {0.890,0.888} | w7 74p {0.889,0.879} | w3 34p
{0.899,0.909}. Baseline 0.742/0.732; v4 GRU 0.869; gate 0.9.

Scoring: P-cap1 CONFIRMED (kc=2, 66 params + 2 states, holds the
top plateau). P-cap2 CONFIRMED (kc=1 at 0.868/0.876 = exactly
v4-class from 32 params + ONE scalar). P-cap3 and the state-vs-
capacity question resolve with a twist: the 34-param STATIC arm
matches the top plateau ({0.899, 0.909}) — at matched small
budgets static ~ or > recurrent. In COMPLETE (Markov)
coordinates, dynamical state confers no material advantage; the
active ingredients are the HARD TRUST REGION + SEQUENTIAL
TRAINING + VAL SELECTION applied to any tiny bounded correction.
The state question's real answer stays where the contamination
dissociation put it: state is required when coordinates are
INCOMPLETE (observable track: static fails at 0.004, recurrence
reaches 0.87+), unnecessary when complete. Everything lands in
[0.87, 0.91] — a new shared plateau, unchased for now (candidates:
val-set size, residual event errors needing the 2b hazard
objective).

M13'S ANSWER, final form: a parallel-trained field (minutes),
the drive's own entrainment (free), and a ~34-parameter bounded
correction (30 min sequential, val-selected) reach F1 0.90+ with
near-perfect f-I and correct rebound — where the correction's
FORM matters only when the coordinates leave state to be
inferred. Dimension x coordinates x dynamics x objective,
measured all the way down.

## OBSERVABLE-TRACK HYBRID declared (2026-09-03, before running)
## — the deployment test

Synthesis adopted (per review): STATE IS PRIMARILY A RESOURCE FOR
RESOLVING PARTIAL OBSERVABILITY — scoped to this system, but now
supported from both directions. The deployment primitive tests it
constructively: deterministic delay buffer (shift register, no
learning) + shared window-field (iid-trained on observable
V-dot, then FROZEN at its known-failed baseline: F1 0.004) +
k in {1, 2} recurrent corrector, trust-bounded on observed V,
sequentially trained with val-F1 selection. Voltage-only
supervision throughout — every ingredient deployment-legitimate.

P-obs1: k=1 rescues the dead formulation to >= 0.80 (a 200x
        event-fidelity delta from one scalar of state).
P-obs2: k=2 reaches v4-class (0.87+) — the compact primitive:
        shared law + buffer + two scalars per neuron.
P-obs3: correction activity concentrates in the drift/decision
        voltage bands (the inspectability signature transfers).
P-obs4: if confirmed, the deployable neuron = ~shared parameters
        (field + rule) + N x k runtime state, k <= 2 — the
        original substrate intuition in its measured final form.

## Observable-track guardrails (2026-09-03, per review; declared
## before scoring)

1. STATIC OBSERVABLE CONTROL declared: delta = eps(V) tanh
   H(window, I), memoryless, matched params, same trust region /
   sequential supervision / selection — closes the loophole that
   the bounded correction NETWORK (not recurrence) rescues B2.
   Outcome tree pinned: static fails + k=1 succeeds -> "same
   information + 1 scalar memory -> large functional rescue"
   becomes a measured architectural statement. k=1 fails but k=2
   succeeds -> minimal observer ~2-dimensional. Both fail ->
   information present but this observer architecture too weak
   (question moves to observer design, NOT information). Success
   with large corrections everywhere -> hesitate: the win must
   show tiny c, bounded delta, band-concentrated activity, field
   carrying bulk dynamics, f-I/rebound preserved — repair of the
   representation, not replacement of the neuron.
2. OBSERVER-STATE DIAGNOSTIC declared: if k=1 works, compare c_t
   against the hidden m, h, n it was never shown — especially
   the excitability margin and the (n up, h down) initiation
   signature. The scalar organizing trajectories by hidden
   excitability = it has spontaneously become an observer state.
3. LANGUAGE CORRECTED, permanently: the delay embedding did not
   "fail" — it SOLVED OBSERVABILITY (D0); the recurrent
   correction solves USABLE STATE RECONSTRUCTION. History stores
   the evidence; a tiny recurrent state turns evidence into
   state. That factorization, if k=1 replicates, is the deepest
   result of the sequence.

## MINI SELECTIVE-SSM CORRECTOR declared (2026-09-03, before
## running) — the parallelism capstone

Design (Mamba's trick, not Mamba's size): c_{t+1} = a_t c_t + b_t
with a_t = exp(-dt / (1 + softplus(w_a x_t))) (input-dependent
forgetting timescale, physically interpretable), b_t = w_b x_t
(input-dependent injection), delta = eps(V) tanh(head(c)) —
~16 SHARED parameters, one scalar state. TRAINING IS FULLY
PARALLEL: under teacher windows the ideal correction is the
frozen field's RESIDUAL r_t = Vdot_true - f_frozen(window, I), a
fixed target sequence; a_t, b_t vectorize; the scan evaluates in
closed form (chunked log-space cumsums); delta regresses onto r_t
with decision-band weighting. No BPTT anywhere. Sequential cost
survives ONLY at deployment: one multiply-add per neuron-step.
Val-F1 rollout selection retained (eval-only sequential).

The A/B/C comparison this completes (same field, trust region,
data, selection, ~matched params): A GRU corrector (sequential
training) vs B mini-SSM (parallel training) vs C static (no
memory). Preregistered readings: SSM ~ GRU >> static -> the
neuron needs a SELECTIVE ACCUMULATOR, not generic nonlinear
recurrence — and it parallel-trains; SSM << GRU -> nonlinear
recurrence is load-bearing and the sequential cost is real;
teacher-window/self-generated distribution shift is the known
risk (the B2 disease) — the trust bound and the accumulator's
reconstruction role are the mitigations under test.
Hyena-style long-convolution noted as a possible later control
(parallel temporal filter vs online state), ranked below the SSM.

## Observable run 1: PROTOCOL FLAW — all arms strangled by a
## bound calibrated for a good base (2026-09-03)

All observable arms (GRU k=1, static, SSM) flat at val-F1
0.01-0.02: the mechanistic trust region (0.01 outside the band)
assumed a GOOD frozen base needing only small corrections; the
observable base is BROKEN (0.004) precisely where the bound is
tightest (spike band). Same-flaw-all-arms — the A/B/C comparison
never started. Runs killed by PID inspection. Fix declared before
relaunch: observable-specific bound eps = 0.05 + 0.25*(V above
deep hyperpolarization) — real correction headroom everywhere
dynamics are active, still bounded well below spike-upstroke
scale (the field must still carry the spikes). Relaunch: GRU k=1,
static, SSM, seeds {0,1}, otherwise identical.

## Observable A/B/C v2 complete (2026-09-03): PERFECTLY FLAT —
## the host determines everything

At eps_hi = 0.3 (patch authority), seeds {0, 1}:
  GRU k=1 (29p, sequential 2130s): {0.005, 0.177}
  SSM k=1 (16p, parallel 160s):    {0.011, 0.168}
  static (33p, sequential):        {0.022, 0.171}
Identical within noise on both seeds; ALL variance is between
seeds (i.e., between the two broken B2 host fields), none between
correction forms. The cleanest possible null: on a nonviable
base, state vs capacity vs selectivity is unmeasurable — a patch
presupposes a host that keeps the representation alive. One
positive extracted: SSM matches GRU at 1/13th the training wall-
clock (the parallel-scan claim demonstrated), pending a setting
where either works. The VM authority grid (eps_hi up to 1.0,
k up to 4, both seeds) is the remaining variant with a mechanism
to escape the host limit.

## Framing correction (2026-09-03, user): the retrofit branch was
## MY drift — the SSM hypothesis needs JOINT placement

Recorded: the user's question was whether selective-scan state can
REPLACE recurrence while preserving parallel training — not
whether a patch can rescue a dead host. The VM sweep answers the
narrower retrofit question (clearly: no). Retrofit branch CLOSED
on sweep completion. The strongest-form experiment declared:

JOINT OBSERVABLE SSM: c_{t+1} = A(x_t) c_t + B(x_t) with x_t =
(teacher window, I) — parallel scan over the whole sequence —
and dV/dt = F(window, c, I) trained on analytic targets over all
timepoints at once. FULLY PARALLEL joint training; state
participates in the dynamics from step one. Deployment:
sequential with self-generated window and carried c.
Arms (same F width/data/val protocol): joint-static baseline
(= B2 v1, already run: {0.007, 0.172}); joint-SSM k {1, 2}
(parallel); joint-GRU k {1, 2} reference (TBPTT through the
c-chain — the sequential price the SSM claims to avoid).
Known risk carried forward honestly: teacher-window training vs
self-generated rollout (the B2 disease) — the hypothesis under
test is precisely that participating state changes the rollout
dynamics class where a static map could not.
P-joint1: SSM k>=1 rescues rollout far above the static baseline.
P-joint2: SSM ~ GRU at equal k — with parallel-scan training
wall-clock << TBPTT (the original claim, in its proper form).

## Joint observable A/B complete; VM stopped (2026-09-03)

joint-SSM k1 {0.243, 0.206} / k2 {0.212, 0.157} (~7 min each,
parallel scan) vs joint-GRU k1 {0.015, 0.182} / k2 {0.193,
0.089} (~13 min each, TBPTT) vs static {0.007, 0.172}.
P-joint2 CONFIRMED: SSM >= GRU throughout, lower variance, half
the wall-clock — selective accumulation beats nonlinear
recurrence in the joint observable setting, parallel-trained.
P-joint1 PARTIAL: participating state lifts the dead seed 30x
(0.007 -> 0.243) but ceilings at ~0.2-0.26 — the teacher-window/
self-rollout mismatch (the knowingly carried risk) caps parallel
training here, exactly as it did for B2. The observable exchange
rate, remeasured at the architectural level: full parallelism
buys ~0.2; every path to 0.87+ in this project has run through
rollout supervision. The mini-Mamba hypothesis lands where the
evidence puts it: the RIGHT state machinery (selective, scannable,
cheap), awaiting the rollout-aware training that would let it
match its GRU-sequential equivalent's ceiling — the hybrid
mostly-parallel + short-rollout-correction recipe now has a
second, sharper target. VM stopped; retrofit sweep banked 13/18
(flat, branch closed).

## Round-7 synthesis + CLOSED-LOOP AGGREGATION declared
## (2026-09-03, per review; before running)

Wording softened per review: "at matched tiny state dimension in
this observable formulation, nonlinear GRU recurrence provides no
detectable advantage over selective affine accumulation" — both
still far from functional. The three-way factorization, each
measured: history contains the information (D0: yes); tiny
selective state can process it (joint SSM: yes, = matched GRU);
teacher-history training transfers to self-generated history
(NO — the isolated bottleneck). The missing ingredient is
CLOSED-LOOP TRAINING DISTRIBUTION, not state capacity. Elevated
lesson: state must PARTICIPATE IN DEFINING the dynamics
F(H, c, I), not arrive as a patch F_bad + delta(c) — state's
value is changing the coordinate system in which dynamics become
easy, the project's through-line confirmed architecturally.

DAGGER-FOR-DYNAMICAL-STATE declared: rounds of (1) PARALLEL
retraining (field iid + SSM scan) on the aggregated corpus;
(2) short rollouts collecting the model's ACTUAL (window, c)
visits, concentrated at the failure frontier (first |V_err| >
10 mV); (3) teacher-labeling those visits (V_dot at matched
times); (4) aggregate, grow horizon, repeat. Sequential cost =
EXPERIENCE COLLECTION only; no gradient ever propagates through
time. v1 scope: corrective pairs train F (the field-with-state);
A/B stay scan-trained on teacher sequences (declared limitation —
full closed-loop A/B correction is v2 if v1 moves).
P-dag1: val-F1 climbs round-over-round (the B2-onpolicy monotone
repair, now with participating state); P-dag2 (the big one): the
aggregation moves ~0.2 materially toward the 0.8+ regime,
demonstrating the residual sequential burden is data collection,
not backprop-through-time. Exchange rate logged per round:
sequential rollout-seconds vs functional gain.

## DAgger v1 scored (2026-09-03): FAILED both seeds — with the
## mechanism visible in the metrics

Rounds (val-F1): s0 {0.197, 0.0, 0.116, 0.002, 0.002, 0.014};
s1 {0.145, 0.057, 0.0, 0.0, 0.034, 0.096}. P-dag1 FALSIFIED:
oscillation, not climb; val-selection banks the round-0 baseline
both seeds. P-dag2 unscored (never left the launchpad).
Diagnostic split captured in s1 round 1: TTD jumped 20x (0.3 ->
6.8 ms) WHILE F1 fell — the corrective pairs (teacher V_dot at
clock-matched times assigned to the model's DRIFTED states) teach
"return to the teacher's flow": they buy stability and blur
decisions, because the labels are dynamically inconsistent with
the states they're attached to. The cost structure worked
perfectly (271 s total, 4.7 s sequential — the DAgger shape is
right); the LABELING is wrong.

v2 declared (next session): PHASE-ALIGNED labeling — pair each
collected model state with the teacher V_dot at the NEAREST
teacher state (within a +-3 ms window), not the clock-matched
one; plus corrective weight annealing. The atlas's phase-aligned
distance instrument built exactly this machinery. If v2 climbs,
the sequential-burden-is-collection claim revives with correct
labels; if not, closed-loop training of the observable SSM needs
more than corrective regression.

## JEPA/EBM program declared + DAgger v2 running (2026-09-03)

Declared for the next tier (per review, ranked): (1) MICRO-JEPA
OBSERVER — train the scalar state to predict REPRESENTATIONS of
future windows (multi-horizon latent targets from a throwaway
target encoder), not instantaneous residuals: the state becomes
"the smallest state that predicts the future dynamical regime."
Parallel scan preserved; encoders are training apparatus only —
deployment stays 1 scalar + ~30 shared params. The OBJECTIVE
LADDER pinned: same SSM/state/field, objectives {residual, raw
future V, multi-horizon V, JEPA latent future} — if #4 wins, the
difficulty was the objective, not the state amount. (2) EBM AS
TEACHER/INSTRUMENT (not as neuron): learn E(state, trajectory-
tube) so dynamically-valid-though-clock-shifted states score low;
use -grad E as the phase-free corrective direction; distill.
Full EBM inference per neuron-step rejected (cost). The house
pattern again: rich expensive measurement -> tiny compiled
mechanism.

DAgger v2 (running): PHASE-ALIGNED labels — each collected model
state paired with the teacher V_dot at the NEAREST teacher window
within +-3 ms of the same sequence (dynamical similarity, not
clock), corrective weight annealed 2.0 -> 0.5. P-dag2v2: the
stability gain of v1 retained WITHOUT the F1 collapse; climb
resumes.

## DAgger v2 scored (2026-09-03): labels fixed, signal weak —
## the objective is now the accused

Phase-aligned rounds: s0 {0.197, 0.168, 0.139, 0.087, 0.099,
0.124} — no crash (v1 hit 0.0), no climb; s1 {0.145, 0.137,
0.188, 0.162, 0.160, 0.157} — the FIRST sustained above-baseline
gain of any closed-loop variant (+0.04 at peak, held 4 rounds).
Verdict: phase alignment eliminates the catastrophic label
inconsistency on both seeds (do-no-harm confirmed) and produces a
weak, seed-dependent genuine improvement. Not the 0.2 -> 0.8
jump; the corrective-REGRESSION signal itself is too weak.
Sequential cost stayed ~10 s/seed — the DAgger cost structure is
proven; what's missing is a stronger definition of what the
scalar state should learn from closed-loop experience. The JEPA
OBJECTIVE LADDER (residual vs raw-future vs multi-horizon vs
latent-future, same SSM/state/field) is promoted to the program's
top open experiment: the accumulated evidence — three weighting
nulls, marker-not-cause, labels-fixed-still-weak — now all points
at the OBJECTIVE as the remaining free variable.

## JEPA OBJECTIVE LADDER declared (2026-09-03, before running)

Question: was the difficulty the amount of state, or WHAT THE
STATE WAS TOLD TO MEAN? Same one-scalar SSM, same field head,
same data and val protocol; the arms differ ONLY in the auxiliary
objective shaping the state (total loss = dV-regression + 1.0 *
L_obj, all parallel over teacher windows):
  A. none (baseline = joint-ssm1: {0.243, 0.206})
  B. RAW FUTURE: predict V_{t+5ms} from (window, c, I)
  C. MULTI-HORIZON: predict V at +5, +10, +20 ms
  D. MICRO-JEPA: predict the LATENT of the +10 ms future window —
     target = frozen random projection of the future window
     (collapse-proof, geometry-preserving); predictor from
     (window, c, I). Training apparatus discarded at deployment;
     the surviving primitive is unchanged (1 scalar + shared
     params).
P-jepa1: C > B > A (longer-horizon prediction forces the state to
carry regime information).
P-jepa2 (the deep one): D >= C — representation-space targets
beat waveform targets because they tolerate phase (the clock/
phase lesson, moved into the objective).
P-jepa3: if D materially exceeds A on ROLLOUT F1, the conclusion
lands: the state's job was regime prediction, not residual
regression — the objective was the bottleneck.
Seeds {0, 1}; rollout eval as always.

## Objective ladder scored (2026-09-03): COMPLETE NULL — and the
## last suspect standing

raw {0.246, 0.201} | multi {0.239, 0.194} | jepa {0.244, 0.200}
| baseline {0.243, 0.206}. P-jepa1/2/3 ALL FALSIFIED: neither
waveform-future nor latent-future targets reshape the scalar
state in any way that survives rollout. The objective joins the
exonerated: state amount, state form, capacity, error magnitude,
error placement, label alignment, and now state-shaping
objectives — every one removed as the cause, by preregistered
experiment.

THE INVARIANT LEFT STANDING, stated as the program's finding:
across every observable-track experiment, the models that reached
0.87+ were trained INSIDE their own rollout regime (v4: TBPTT on
self-generated states); every model trained on teacher-regime
data — whatever its state, capacity, labels, or objective —
ceilings at ~0.2. The gap is not about what the model is or what
it is told; it is about WHICH DISTRIBUTION IT EXPERIENCES DURING
LEARNING while coupled to its own dynamics.

Final declared variant (v3, next session): invert DAgger — make
the model's OWN long rollouts the primary training distribution
(cheap to generate, no gradients), phase-aligned teacher labels,
teacher data as the minority anchor. If that also fails, the
conclusion is earned in full: closed-loop event fidelity at this
scale requires learning-while-coupled (sequential gradients), and
the parallel-training boundary is drawn exactly there — a sharp,
falsifiable, and honestly-reached edge for the whole program.

## Wording correction + three levels + CTM layer parked
## (2026-09-03, per review)

Round-9 conclusion SCOPED: "across the TESTED observable
architectures, state forms, capacities and objectives, the
persistent predictor of success is whether training occurred
within the model's own rollout regime" — an untested architecture
could still break the pattern. Refined conclusion adopted: the
meaning of state is inseparable from the closed-loop dynamics in
which it will be used; c -> V -> H -> c is a loop, and one side
cannot be fully learned while pretending the other stays
teacher-generated. Why DAgger helped only weakly: it always
learns from the PREVIOUS model's rollout distribution.

THREE LEVELS pinned: (1) teacher distribution — exhaustively
ruled out; (2) model distribution collected offline — the
INVERSION tests this properly; (3) current-model distribution
while parameters change (contemporaneous coupling) — implicated
only if (2) fails. Either outcome names what the sequential core
buys: closed-loop EXPERIENCE (if inversion works: "experience
necessary, gradients-through-time not") or contemporaneous CREDIT
ASSIGNMENT (if it fails and short current-model BPTT succeeds).

CTM LAYER declared (not run): internal tick-time as a second time
axis; the geometrization move — represent internal trajectories
in a mode basis z_i(tau) = sum_r a_ir phi_r(tau) so
synchronization S_ij = a_i^T G a_j becomes geometry, computable
without marching ticks; dynamic role assignment (same unit,
different coalition per phase) as conditional effective
connectivity W_eff = W g(r_i, r_j) — the computational graph as a
dynamical object. Next conceptual layer AFTER the closed-loop
question resolves; premature for one neuron.

## INVERTED DAGGER declared (before running)

One variable only: the TRAINING DISTRIBUTION. Rounds of
{generate large model-rollout trajectory corpus with the CURRENT
model (gradient-free); phase-align teacher labels per step;
retrain BOTH the scan (A/B) and the field on a batch mixture} —
rollout sequences train identically to teacher sequences (scan +
vdot loss), so the state dynamics themselves adapt to the model
distribution, unlike DAgger v1/v2 (F-only). Sweep p_rollout in
{0.5, 0.8, 0.95, 1.0}, seeds {0, 1}; 3 aggregation rounds; same
scalar SSM, field, phase matcher, optimizer.
P-inv1: F1 rises with p_rollout.
P-inv2: flat ~0.2 through p=1.0 -> level two ruled out; the
irreducible ingredient is contemporaneous coupling, and the
parallel-training boundary is drawn there — precisely.

## INVERSION SCORED (2026-09-03): LEVEL TWO RULED OUT — the
## boundary is contemporaneous coupling

Best-val by p_rollout (seeds 0/1): 0.5 {0.201, 0.170} | 0.8
{0.201, 0.175} | 0.95 {0.201, 0.144} | 1.0 {0.201, 0.148} — at
every dose, best ~ round-0 baseline; within rounds, rollout-heavy
training COLLAPSES (round 1 injections drive seed 0 to ~0.002-
0.02). P-inv1 FALSIFIED: F1 does not rise with own-distribution
fraction; it falls. Level two — offline training on the model's
own distribution, even at 100%, with adapting state dynamics and
phase-aligned labels — is ruled out.

One recurring micro-signal, noted: seed 1 shows a small (+0.03)
round-1 bump at EVERY dose before collapse — one round of FRESH
own-distribution data helps slightly; aggregated STALE data (from
previous parameter settings) harms. The staleness gradient itself
supports the level-three reading from inside the data.

THE BOUNDARY, drawn precisely and earned by ten rounds of
preregistered experiments: state propagation, field fitting,
representation learning, corrective optimization, and even
own-distribution data GENERATION all parallelize; what does not —
by any of the nine mechanisms tested — is the FEEDBACK BETWEEN
PARAMETER UPDATES AND THE DISTRIBUTION THOSE PARAMETERS INDUCE.
Contemporaneous credit assignment through the closed loop is what
the ~0.87 sequential models buy. The scoped conclusion: for the
tested observable formulations, closed-loop event fidelity
requires learning while coupled; the exchange-rate question
(how LITTLE contemporaneous coupling suffices — the 99%-parallel
+ brief current-model-BPTT experiment) is the declared successor,
alongside the parked CTM layer.

## Endgame plan + two experiments declared (2026-09-03)

ADOPTED: paper write-up is the deliverable; the exchange-rate
experiment is the one result that strengthens the ending without
changing the story; CTM = M14 (the handoff: M13 asks what
irreducible state/coupling a dynamical unit needs; M14 asks
whether tick-space + synchronization geometry lets populations
reorganize computation). Paper wording pinned: no "cannot" —
"within the tested formulations and regimes, contemporaneous
closed-loop training was the only condition associated with high
event fidelity, after extensive controls." Narrative = the eight
isolated questions, not chronology.

EXCHANGE-RATE EXPERIMENT declared: mostly-parallel training
(scan + iid on teacher data) with a fraction p of optimizer
updates replaced by COUPLED updates — short (5 ms) current-model
BPTT segments from teacher-primed starts, loss on the model's own
rollout vs teacher. Sweep p in {0, 0.02, 0.05, 0.15, 0.5, 1.0},
seeds {0, 1}. The curve F1(p) is the signature figure candidate:
"temporal credit necessary, but only sparsely" if a small p
recovers most of the gap.

LOOKAHEAD declared (user proposal; fixed-lag smoothing): can one
sample of future observation substitute for hidden state?
Stage 1 (zero training, runs first): the decision-conditioned
collision diagnostic on H_t vs H_t + V_{t+1} vs H_t + forward
difference — does one-step lookahead collapse decision-band
ambiguity? Only if yes: (A) causal SSM vs (B) one-tick-delayed
smoother vs (C) predictor-corrector internal tick (the autonomous
analogue — no true future exists for a standalone neuron; the
internal tick asks "what does my provisional next step imply
about my current state?" — the CTM bridge in miniature).
Principle at stake: latency <-> state complexity, a hardware-
grade trade.

## Lookahead stage 1 (2026-09-03): NULL — stopped per rule

Decision-band collision: H 0.0178 | H+V(t+1) 0.0170 | H+dV+
0.0178. The window's decision ambiguity is already at floor (E2
redux); one-step lookahead has nothing to collapse. The latency-
for-state trade has no purchase in this system because
INFORMATION was never the deficit — the branch stops at stage 1
as declared. Predictor-corrector/internal-tick survives as an
M14 concept only.

## Exchange-rate watch-list (2026-09-03, per review; recorded
## while the sweep runs)

1. Primary axis = SECONDS of coupled optimization (schedule-
   independent, generalizable), not fraction p; both are logged
   per run (coupled_seconds, n_coupled).
2. Current sweep uses UNIFORM interleaving (recorded). If the
   curve has a knee, the conditional follow-up sweeps SCHEDULE at
   fixed p: uniform vs early-concentrated vs late vs after-
   parallel-convergence — discriminating basin-INITIALIZATION
   (coupling moves the model into the right basin once, parallel
   training keeps it there -> the sequential core is even less
   fundamental) from continuous CALIBRATION (performance decays
   when coupling stops).
3. NON-MONOTONICITY watch: composition history predicts p=0.02/
   0.05 may beat p=0.5/1.0. If so, the conclusion upgrades:
   parallel local supervision and coupled temporal credit are
   COMPLEMENTARY (local physics + consequence-aware credit), not
   approximations of each other — the hybrid recipe would be
   conceptually cleaner than full BPTT, not merely cheaper.
4. The lookahead null's role in the paper: removes the "just
   needed a bit more information" escape hatch — the bottleneck
   was never information content.

## Primitive roadmap + timescale maps declared (2026-09-03)

PRIMITIVE PATH recorded (per review): pretrain once -> freeze/
share -> train structures around it. Phase A: fully-frozen
M13Cell, train connectivity only — the "useful primitive vs good
impersonator" test (vs LIF/AdEx/Izhikevich at matched state/
params/gates on temporal tasks). Phase B: shared law + tiny
per-population code c_i (1-4 numbers: cell types). Scaling risk
named: in-network input distributions (coupled, correlated,
oscillatory) differ from training drives — scale 1 -> 2 -> 8 ->
32, with the TWO-NEURON coupling sweep (synchronize/destabilize/
sensible regimes) as the first composition test. Training/
deployment separation: rich differentiable teacher form for
network-gradient purposes, compiled tiny cell for hardware. The
stack: primitive -> connectivity -> synchronization/routing (M14)
-> task.

TIMESCALE MAPS declared (two zero-cost instruments, run now):
T1: HH local Jacobian spectrum — tau_i(x) = 1/|Re lambda_i| by
    voltage band; check against the measured fingerprints (12 ms
    sufficiency window, 5.4 ms entrainment, slow n/h drift,
    fast-m).
T2: learned SSM effective clock — tau_SSM(t) = -dt/log a_t by
    band; prediction: short memory where input determines state,
    long memory in the ISI where excitability must be
    reconstructed — one state with a STATE-DEPENDENT CLOCK.

## Timescale maps (2026-09-03): the physics has a state-dependent
## clock; the learned primitive's clock NEVER WOKE

T1 (HH Jacobian, median ms fast->slow by band): hyperpol {0.15,
0.30, 4.4, 7.8} | subthreshold {0.21, 4.9, 5.2, 8.4} | decision
{0.05, 1.0, 2.5, 4.4} | spike {0.04, 0.24, 0.69, 1.05}.
Fingerprints validated: 5.4 ms entrainment ~ the ~5 ms
contracting mode; 12 ms sufficiency window ~ 1.5x the slowest
mode (8.4 ms); decision-band fast-mode collapse (0.05 ms) = the
initiation instability. Effective timescale structure varies
FOURFOLD across regimes — "how many states" refines again: the
active timescales are state-dependent.

T2 (learned SSM effective clock): FLAT at tau ~ 1.6 ms in every
band (p10-p90 ~ 0.1 ms) — which equals the INITIALIZATION value
(1 + softplus(0)). The selective gate never trained: no
non-coupled objective ever pressured a_t to vary. The SSM ran as
a fixed leaky trace with dormant selectivity — retro-explaining
SSM ~ GRU ~ static from a new angle, and adding a candidate
mechanism to the boundary story: perhaps CONTEMPORANEOUS COUPLING
is what trains the clock.

DECLARED CHECK: when the exchange-rate sweep lands, rerun T2 on
the p=0.5 / p=1.0 checkpoints vs p=0 — if coupled training wakes
the clock (tau map becomes state-dependent, ideally tracking
T1's structure), the boundary result gains its mechanism: coupled
gradients are the only signal that reaches the temporal
selectivity parameters.
