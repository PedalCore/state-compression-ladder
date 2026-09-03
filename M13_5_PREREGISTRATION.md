# M13.5 — IPC-per-hardware-cost: preregistration

Status: **draft for reviewer sign-off, pre-implementation.**
Nothing in this protocol has been run. v1/v2 harnesses were
discarded for two confounds (isolated-cell measurement; asymmetric
input nonlinearity); this document exists so a reviewer can say
"if M13 wins this, the win belongs to its dynamics."

## Hypothesis

At matched implementation cost, does a richer dynamical neuron
provide more useful linear AND nonlinear temporal processing
capacity than simpler dynamical primitives?

Kept as two SEPARATE questions, measured in order:
- **A. Intrinsic capacity** — at equal cell count / exposed
  outputs, what computation does each primitive provide?
- **B. Hardware efficiency** — at ~equal runtime cost, which
  provides more? (A first; cost-normalise for B. Hardware
  matching is NOT enforced inside the A measurement.)

## The four non-negotiables (reviewer's sign-off conditions)

1. **Identical affine input path.** Canonical scalar iid input
   `u_t ~ U[-1,1]`. The ONLY permitted input transform is affine:
   `I_i,t = I0 + g_in*b_i*u_t + g_rec*sum_j W_ij*y_j,t`, with
   `b_i`, `W_ij` frozen random masks SHARED across all arms. No
   tanh, no cell-specific encoder, no preprocessing net. If a
   primitive needs a nonlinearity to make nonlinear IPC, the
   primitive must provide it. (A pure linear trace bank therefore
   MUST show ~0 nonlinear IPC — that is a correct control, not a
   failure.)
2. **Identical recurrent reservoir.** Real recurrent reservoir;
   same N, topology, sparsity, connection signs, input mask,
   sequence, sampling interval for every arm. W generated once per
   topology seed and REUSED identically across cell types.
   Reservoir never task-trained — only a linear readout trains.
3. **Identical exposed output.** Primary benchmark exposes ONE
   membrane-like scalar V_i per cell (LIF, AdEx, M13 all have V).
   No giving M13 all hidden variables while LIF gets a spike bit —
   that measures interface width, not dynamics. Spike-only output
   is a secondary benchmark; any spike filtering is identical
   across arms AND counted as runtime cost.
4. **Preregistered equal-budget operating point + full cost
   ledger.** Every arm gets the SAME fixed tuning budget (a
   g_in x g_rec [x I0] grid, e.g. 7x7); pick (g*,g_rec*) on a
   calibration sequence, freeze, evaluate IPC on NEW sequences.
   Publish the whole gain heatmap, not just the winner. Runtime
   cost counted PER PHYSICAL SAMPLE (not per Euler substep):
   live state scalars/bits, mults, adds, comparisons, exp,
   tanh/LUT, parameter bits, integration substeps, recurrent-edge
   ops. All deployed M13 arithmetic counts (the frozen base field,
   not just the 32-param corrector); shared weight bits amortise
   as bits/N but per-neuron arithmetic does not vanish.
   Report C_cell and C_connectivity SEPARATELY.

## IPC measurement

Standard orthogonal-polynomial (Legendre) construction, identical
target library for all arms. Report separately:
`C1 (linear), C2, C3, C4, C_total`.
- **Null calibration (required):** for every target also fit the
  readout to a shuffled target; count only capacity materially
  above the finite-sample null. Use held-out `C_k = max(0,
  R2_heldout)` after null correction. Guard: `C_total >> N` (with
  N exposed scalars) is a harness red flag.
- **Integration fairness:** all cells receive input at the same
  external interval; internal substeps allowed but counted;
  one-time qualification that each arm's IPC is stable to halving
  its step (compare artifact vs dynamics).

## Arms

Qualification first — THREE arms only:
`linear trace bank | LIF | M13 kc=1`.
Expected sanity pattern: trace = memory, ~0 nonlinear IPC; LIF =
event nonlinearity; M13 = richer nonlinear. If that behaves and
survives inspection, ADD: AdEx/Izhikevich, M13 kc=8 (ceiling),
and the key control below.
- **Multi-timescale linear bank** (the deep control): same state-
  scalar count as the comparison, log-spaced timescales. Tests
  "rich nonlinear geometry" vs "simply a better memory spectrum"
  (motivated by M13's dormant-clock finding). If a 4-timescale
  linear bank matches M13, hugely informative; if not, the HH
  nonlinearity earns its keep.
- Generic nonlinear baseline, IF wanted, is a NAMED primitive
  (e.g. `c_{t+1}=a c_t + b*tanh(u_t)`) with the tanh counted in
  its cost — never a quiet encoder on one arm.

## Seeds (locked before results)

8 frozen topology seeds; identical topologies reused across arms
(paired comparison — same reservoir realisation across cell types
is far more powerful than independent draws); 2 input sequences
per topology; fixed train/calibration/test lengths; one
preregistered gain grid; median + per-seed reported.

## Falsification criterion (written before running)

M13 is NOT computationally advantageous merely because raw IPC is
larger. The useful result requires EITHER
`IPC_M13 / cost_M13 > IPC_baseline / cost_baseline`
OR M13 reaching a target IPC/accuracy with materially fewer
cells/connections. This blocks the trap "a more complicated
neuron unsurprisingly computes more." The real question: does the
extra local complexity pay for itself?

## Result tables (two, kept separate)

1. Intrinsic: primitive | states | C1 | C2 | C3 | C4 | C_total
2. Efficiency: primitive | IPC/state-bit | IPC/mult | IPC/gate-cost

## Process

preregister (this doc) -> reviewer sign-off on the four non-
negotiables -> implement v3 -> smoke controls (trace~0 nonlinear;
C_total not >> N; step-halving stable) -> production. No manual
rescue runs; no per-arm hand-tuning outside the shared grid.

---

# AMENDMENTS (reviewer conditional sign-off, 2026-09-03) — FROZEN

Reviewer verdict: **approved in principle for M13.5-A / harness
qualification**, subject to the eight points below being frozen
before implementation. The final "IPC per hardware cost"
conclusion is NOT signed off until B1's cost metric is frozen;
gate/energy efficiency is reserved for a later synthesis protocol
(B2). All loose phrases below are now formulas/deterministic rules.

## A1. Recurrent observable — exact

Voltage-based cells feed the recurrent net and readout the SAME
normalized observable: `y_i = (V_i + 65 mV) / 100 mV`, constants
identical across all voltage arms (raw V never enters W — its DC
offset would make a model-dependent recurrent bias). Trace/
control arms define their dimensionless state directly as `y_i`.
Readout may standardize features by TRAINING-SET mean/std, but
that standardization NEVER feeds back into the reservoir. The
linear trace arm contains NO clipping, saturation, or nonlinear
normalization anywhere (else the "linear null" acquires nonlinear
IPC — the v2 failure mode).

## A2. IPC target library — deterministic

Orders `d in {1,2,3,4}`; single-tap delays `tau in {1..L}`,
`L = 20`; **tau=0 EXCLUDED from the primary score** (direct
input sensitivity is not memory/computation). Cross-delay
products: for orders 2 and 3, include all delay pairs/triples
drawn from a fixed deterministic list (seed-0 enumeration of
`tau_a < tau_b [< tau_c] <= L`, capped at 200 order-2 and 100
order-3 targets, generated once and frozen). Rename the aggregate
`C_total -> C_{<=4,L}`: the capacity in THIS finite delay/order
library, not a theoretical total.

## A3. Readout estimator — locked

washout `1000`; readout-train length `15000`; held-out test
length `8000` (fresh sequence). Features: training-set centering
+ unit-variance scaling; explicit intercept column. Estimator:
ridge; `lambda` chosen ONCE per arm by validation on a held-out
slice of the CALIBRATION sequence over `{1e-8,1e-7,...,1e-2}`,
then frozen; `rcond = 1e-12` in the solve. Identical for all arms.

## A4. Null rule — mathematical

For each target q, build a null from `20` fixed circular shifts
of that target; `C_q^primary = max(0, R2_q_heldout)` iff
`R2_q_heldout > Q_0.95(R2_null)`, else 0. Report BOTH raw
held-out IPC (literature-comparable) and null-thresholded IPC
(primary robust measurement).

## A5. Operating-point selection — locked objective

`(I0, g_in, g_rec)* = argmax C_{<=4,L}^calibration`, then frozen;
IPC computed on NEW test sequences. Grids (equal budget, every
arm): `g_in in logspace(-1, 1, 7)`, `g_rec in {0, 0.1, 0.3, 0.5,
0.7, 0.9, 1.1}`, `I0 in {-0.5, 0, 0.5}` (spiking arms only; 0 for
continuous). Divergence = any |state| > 1e3 or non-finite over
the calibration run -> point marked invalid, excluded, never
rescued. Tie-break: smallest g_rec, then smallest g_in. Calibration
sequence is a SEPARATE realization from the IPC test sequence. No
inspecting individual C_d to pick a "better-looking" point.

## A6. Baseline fairness — later phase

Qualification: canonical fixed LIF is fine. FINAL comparison adds
BOTH a labelled canonical LIF AND an optimized-LIF given the SAME
preregistered candidate-configuration budget as M13's operating-
point search (equal number of candidate configs, not equal
tunable params). Same for AdEx. This blocks "M13 was pretrained
while LIF's tau/thr/reset were arbitrary."

## A7. M13 state count — no shorthand

FORBIDDEN anywhere in M13.5: "one-state M13". kc=1 is one
CORRECTION state; the deployed runtime primitive also carries the
mechanistic field state. Intrinsic table states column reads
`M13-kc1 = 4+1`, `M13-kc8 = 4+8`. Hardware ledger includes the
ENTIRE frozen field evaluation and its integrator substeps, not
just the corrector's 32 params.

## A8. Statistical unit — no pseudoreplication

Average the 2 input-sequence results WITHIN each topology seed;
treat the `8` topology seeds as `n=8` PAIRED units (same reservoir
realization across cell types). Report all paired points; paired
bootstrap CI over topology seeds for every difference/ratio.

## B split — algorithmic proxy now, synthesis later

**B1 (preregisterable now):** states/sample, state bits,
parameter bits, mults, adds, LUT/nonlinear ops, integration
substeps, edge MACs — all PER PHYSICAL SAMPLE, all deployed M13
arithmetic counted. This is the hardware proxy for M13.5.
**B2 (reserved):** synthesized area/pJ under a fixed requirement
("advance all N cells one physical step every T_s") at fixed
precision + synthesis flow — a later protocol, NOT this paper.
"IPC/gate-cost" is NOT a primary M13.5 result (gate cost depends
on precision, time-multiplex-vs-replicate, clock, library — none
fixed yet).

## Passage-to-production gates (all must pass; else STOP and find
## the harness source, do NOT tune until it looks right)

1. linear-trace nonlinear C_{2:4} statistically indistinguishable
   from its null;
2. C_{<=4,L} within finite-sample tolerance of the feature-
   dimension bound (not >> N exposed scalars);
3. halving the integration step does not materially reorder arms;
4. gain maps show a genuine stable operating REGION, not a single
   pathological optimum;
5. an independent implementation reproduces one small IPC case
   analytically/numerically.

## g_rec=0 diagnostic panel (before production)

Run a zero-recurrence slice as a diagnostic (not a headline arm):
separates single-cell temporal computation from computation
CREATED by recurrent interaction. If M13's nonlinear-IPC advantage
is already present at g_rec=0, the substrate does the work; if it
appears only with coupling, the resource is the interaction of
local dynamics with recurrence. Central to the original question.

## What a convincing M13 result looks like (preregistered shape)

NOT `C_M13 > C_LIF` alone. The target signature:
`C1_M13 ~= C1_LIF` (comparable linear memory) BUT
`C_{2:4}_M13 >> C_{2:4}_LIF` (more nonlinear basis), AND
`C_{<=4,L}_M13 / cost_M13 > C_{<=4,L}_LIF / cost_LIF`.
That says the richer state turns history into useful nonlinear
basis functions efficiently, not merely stores more of it.
Falsifier retained: if the multi-timescale LINEAR bank matches
M13 after state/cost normalization, the resource was the
TIMESCALE SPECTRUM, not HH nonlinear geometry — still an
excellent result.

## Status after amendments

M13.5-A / harness qualification: **signed off** with the above
frozen. Qualification arms remain THREE (linear trace | LIF |
M13-kc1). v3 may be implemented against this spec; production
gated on the five passage tests. Final efficiency conclusion
gated on B1 cost metric being frozen; B2 synthesis is future work.

---

# QUALIFICATION OUTCOME (2026-09-03) — HALT FOR REVIEWER RULING

The v3 harness was built to this spec and run in qualification.
**Integrity gates pass** (linear-trace nonlinear IPC = 0.007 ~=
null; C_{<=4,L} <= N; LIF step-halving-stable). **The spiking-
substrate step-halving gate FAILS structurally**: an instantaneous
membrane-snapshot observable at a fixed shared sample interval
cannot be simultaneously (i) step-halving-stable and (ii) able to
expose HH's spike nonlinearity — coarse sampling aliases the spike
(IPC never dt-converges: C1 0.08->0.80), fine sampling is dt-stable
but membrane-low-passes the fast input to ~0 IPC, and a low-pass
observable stabilises by smoothing the nonlinearity away. Leaky-
integrator arms pass trivially because they have no stiff event.

This is exactly the harness flaw qualification exists to catch,
found before any headline number was trusted. Fixing it (input-
hold/masking; a windowed observable applied identically to every
arm; a nonlinear-capacity operating-point objective; a stated HH
substep floor folded into the B1 cost) changes what "IPC of HH"
means — a preregistration decision, not a mid-run tuning knob.
Full diagnostic + four candidate resolutions in M13_5-IPC.md.
**No production IPC numbers exist. M13.5-A is NOT qualified until
the spiking-observability question is resolved in writing.**

---

# AMENDMENT II (2026-09-03) — reviewer ruling on the qualification
# halt. FROZEN before v3-production. Approve 1+2+4; 3 in modified
# (two-operating-point) form.

## II.1 Symbol-held common input (hold chosen by convergence, not IPC)

Input is piecewise-constant per symbol: `u(t) = u_k` for
`k*Th <= t < (k+1)*Th`, `u_k ~ U[-1,1]` iid. EVERY primitive gets
the SAME `Th`. `Th` is chosen from the candidate set
`Th in {0.5, 1, 2, 5} ms` as the SMALLEST value passing a
convergence criterion evaluated WITHOUT looking at IPC:
- run identical input under `dt` and `dt/2`; require the raw
  exposed window statistic `y_i[k]` (II.2) to change by
  `< tol_stat = 2%` (relative RMS over cells and symbols);
- for spiking arms additionally require per-cell event count
  agreement within `<= 5%` and mean spike-time shift `< 0.1 ms`.
`Th* = smallest common hold giving converged dynamics for ALL
qualification arms`. If no candidate passes, qualification FAILS
(no production). IPC delays are SYMBOL delays `tau = 1..L`;
physical memory horizon `t_memory = tau * Th*` (reported, and used
when comparing slower vs faster primitives).

## II.2 Common LINEAR window-average observable (primary)

Primary observable for ALL arms:
`y_i[k] = (1/Th) * integral_{kTh}^{(k+1)Th} z_i(t) dt`, where
`z_i(t)` is the ONE exposed scalar: affine-normalized `V_i` for
voltage cells (identical constants), the scalar state for the
trace bank. A window MEAN is a linear observation operator and
cannot manufacture nonlinear IPC — this is why it, not spike
count, is the cross-arm primary. Spike count is a SECONDARY
spiking-only analysis, never the headline. The window accumulator/
register is counted in the B1 ledger.
CRITICAL: the window average is ONLY what the linear readout sees.
It is NOT fed back into the reservoir and is NOT extra reservoir
state. Recurrent coupling stays instantaneous and one-scalar for
every arm during the hold:
`I_i(t) = I0 + g_in*b_i*u_k + g_rec * sum_j W_ij * z_j(t)`.
The measurement layer must not change the dynamics.

## II.3 TWO preregistered operating points (replaces single argmax)

On the calibration set, freeze BOTH:
`theta_total* = argmax_theta C_{<=4,L}` (general-purpose capacity)
`theta_NL*    = argmax_theta (C2+C3+C4)` (nonlinear feature-gen),
`theta = (I0, g_in, g_rec)`. Evaluate BOTH once on test. Report
both, AND the full gain-map Pareto front of `C1 vs C_{2:4}`.
Rationale: argmax-total alone can pick a subthreshold linear
regime; argmax-NL alone would deliberately flatter spikers.
Reporting both answers two distinct questions and lets a
subthreshold total-capacity optimum stand as a RESULT, not be
forced away.

## II.4 Substep + physical-time cost in B1 (not normalized away)

If preserving richer dynamics needs `n_substep` internal field
evaluations per input symbol while LIF needs one, those
evaluations ARE part of the price. Per physical input symbol,
ledger: `n_substep`, all field evals, adds, mults, nonlinear ops,
recurrent-edge ops, state bits, window accumulator. Report BOTH
`IPC / input_symbol` AND `IPC / physical_second = (IPC/symbol)/Th*`
so a long hold cannot buy free capacity by letting a rich
substrate compute longer per symbol. B1 primary ratios now:
`IPC/operation, IPC/state_bit, IPC/physical_time`.

## II.5 Diagnosis wording (softened before publication)

The v3 finding is NOT "a stiff spiker cannot be observably stable."
It is: **under the instantaneous-snapshot protocol and the tested
integration scheme, the HH-like substrate did not yield an
observable simultaneously numerically converged AND informative at
the shared sampling interval.** The general lesson —
`equal sample clock != equal observation of dynamical computation`
when primitives have very different dynamical bandwidths — is the
real scientific content. Snapshot IPC is not a neutral measurement
interface across bandwidths; a common symbol timescale + linear
temporal observation + explicit physical-time cost is the
defensible comparison.

## II.6 Revised qualification gates (ALL must pass to sign off prod)

1. linear-trace nonlinear capacity remains at null;
2. `C_{<=4,L}` respects the feature-dimension bound;
3. the raw dynamical / window observable is step-halving converged
   (this is the II.1 selection criterion);
4. IPC itself is subsequently step-halving stable;
5. LIF AND M13/HH both possess non-pathological regions in the
   common gain grid;
6. no production statistic is taken from an unstable trajectory.
Passing all six -> production signed off.
