# M13.5 P2 — task-based temporal computation per hardware cost
# Preregistration (draft for reviewer sign-off; nothing run)

## P1 closed as a negative result (do not rescue)

**Passive linear-readout IPC is not a stable cross-substrate
capacity probe for the tested stiff spiking dynamics, even after
symbol holding and linear window observation.** Analytic HH and
the deployed M13 surrogate (field, and kc=1 = field+trust
corrector) both fail step-halving convergence of the per-symbol
observable (rel-RMS 44-88%) and yield near-null, unstable IPC.
The important content is the opposite of "the learned field is
numerically bad": **M13 faithfully preserves the near-threshold
event structure that makes passive IPC ill-conditioned.** Smoothing
the vector-field REPRESENTATION did not smooth away the
computationally relevant spike-event geometry — the corrector made
convergence marginally worse, not better. This falsifies the
"compression regularizes observability" hypothesis. Passive IPC is
retained ONLY for smooth/rate substrates; spiking arms are excluded
with this as the cited reason. P2 replaces it for the spiking arms.

## Why P2 is the better question

Passive IPC asked "how much abstract basis-function capacity exists
in this substrate?" P2 asks the hardware-relevant question closer
to the founding thesis: **given the actual event dynamics this
substrate produces, does it solve temporal computations with fewer
resources?** Targets are defined at SYMBOL level, so the score does
not care whether a spike landed at 4.98 or 5.02 ms.

## Two orthogonal tasks (symbol-level targets)

Symbol stream `u_k ~ U{-1,+1}` (Bernoulli +/-1) for parity;
`u_k ~ U[-1,1]` for recall. Delay ladder (both tasks):
`d in {1, 2, 4, 8, 16}` symbols.

1. **Delayed recall (MEMORY axis):** `y_k = u_{k-d}`. Linear
   readout; score = held-out R^2 (or NMSE). A linear trace bank
   CAN solve this — it is the memory control.
2. **Delayed parity / XOR (MEMORY + NONLINEARITY axis):**
   `y_k = sign(u_{k-d1} * u_{k-d2})` (2-bit) and k-bit parity over
   selected delays. Logistic readout; score = held-out accuracy.
   A linear trace bank CANNOT solve this without nonlinear
   dynamics — this is the discriminating task.

## Reservoir, input, interface (carried from Amendment II, intact)

- Same topology `W` + input mask `b` per seed, shared across arms;
  same symbol hold `Th`; strictly-affine input path
  `I_i(t) = I0 + g_in*b_i*u_k + g_rec*sum_j W_ij*z_j(t)`; one
  exposed scalar `z_i` per cell (affine-normalized V / trace state).
- **Interface observable (now explicitly part of the deployed
  interface, not a capacity probe):** window accumulator
  `r_i[k] = (1/Th) integral_{kTh}^{(k+1)Th} z_i(t) dt`, applied
  IDENTICALLY to every arm and COUNTED in the B1 cost ledger. This
  is no longer pretending to measure intrinsic IPC — it is the
  readout front-end of the deployed system.
- Readout: recall = ridge linear; parity = logistic (L2). Trained
  on `r[k]` -> `y_k`. Readout is the ONLY trained component; the
  dynamics stay frozen.

## Arms

First pass (qualification of the P2 harness): `linear-trace |
LIF | M13-kc1`. If sane, add `AdEx | M13-kc8` and (labelled)
`optimized-LIF` at equal candidate-config budget (Amendment A6).

## Operating point — no tuning on test

Per (task, arm): same grid `(I0, g_in, g_rec)` and same
candidate-config budget for every arm; select `theta*` on a
VALIDATION split by task score; LOCK; evaluate ONCE on a held-out
test split with fresh sequences. Separate calibration / validation
/ test sequences. Divergence -> config invalid, excluded. No arm
is tuned on test accuracy.

## Cost ledger (B1, exactly as already frozen)

Per physical input symbol: `n_substep`, all field/corrector
arithmetic (mults, adds, nonlinear ops), state bits, recurrent-edge
ops, AND the window accumulator/readout interface cost. Report task
score vs cost as: **score / operation, score / state-bit,
score / physical-second** (physical-second via `Th`), so a long
hold cannot buy free advantage.

## Convergence gate (the P1 lesson, reused)

The TASK SCORE (not the microscopic trajectory) must be
step-halving stable: recompute the locked-config score under field
substeps `n` and `2n`; require the score to change by
`< tol_score = 3%` (absolute for accuracy, relative for R^2). This
is the gate-4 analogue and the whole reason task-based should work
where passive IPC did not — the score is an expectation robust to
spike-time jitter. **If the score is ALSO dt-unstable, P2 is
likewise ill-posed for spikers; report that and fall back to
P3-only (passive IPC for smooth substrates, spikers documented as
excluded).**

## Statistical unit (Amendment A8)

8 paired topology seeds (same reservoir realization across arms);
average within-seed replicate sequences; paired bootstrap CI over
seeds for every arm-vs-arm score and score/cost ratio.

## Pre-committed interpretation tree

- **M13 wins recall only** -> richer MEMORY.
- **M13 wins parity but not recall** -> richer NONLINEAR temporal
  processing.
- **M13 wins both** (after cost normalization) -> broad
  computational advantage — the founding thesis, on well-posed
  footing.
- **LIF matches M13 after cost normalization** -> richer substrate
  does NOT pay for itself.
- **Trace bank matches recall but fails parity** -> harness
  behaving sensibly (a required sanity signature, not a result).
- **Task score dt-unstable** -> P2 ill-posed for spikers -> P3.

## Falsification

M13 is not advantageous merely by higher raw accuracy: the claim
requires `score_M13 / cost_M13 > score_LIF / cost_LIF` on at least
the parity axis, OR M13 reaching a target accuracy with materially
fewer cells/edges. Same anti-"a bigger neuron computes more" guard
as before.

## Status

Draft. Requires reviewer sign-off before the harness runs. Cost
metric for the final efficiency claim (B1) already frozen; B2
synthesis remains future work.

---

# P2 FREEZES (reviewer conditional sign-off, 2026-09-03) — FROZEN
# before the convergence-only run.

## F1. Null-adjusted skill metric (never raw accuracy/cost)
`S_recall = max(0, R^2)`; for balanced binary parity
`S_parity = max(0, 2*(accuracy - 0.5))`. Cost-normalize S, not raw
accuracy (a useless 0.5 classifier must score 0). PRIMARY hardware
result is a performance-cost FRONTIER — "cost to reach S=0.8" —
not S/ops, which is easier to game.

## F2. Deterministic parity target (no post-hoc delay tuples)
2-bit sign parity `y_k = sign(u_{k-d} * u_{k-2d})`,
`d in {1,2,4,8,16}`. 3-/4-bit parity is a LATER difficulty
extension, delays fixed in advance.

## F3. Absolute convergence criterion
Recall: `|Delta R^2| < 0.03`; parity: `|Delta accuracy| < 0.03`,
between substeps n and 2n, with IDENTICAL input sequence, topology,
initial condition, operating point, train/val/test split and
readout procedure. Object of the gate (the P2 hypothesis): does the
FUNCTIONAL OUTPUT converge even when individual event windows do
NOT? Internal feature values are NOT required to converge (P1
showed they don't).

## F4. Minimal convergence gate (no full ladder, no grid search)
Arms trace | LIF | M13-kc1, identical paired seeds + data. Only
`d = 2` and `d = 8`, on delayed-recall AND 2-bit delayed-parity, at
substeps n vs 2n. Single preregistered representative operating
point `(I0,g_in,g_rec) = (0.0, 1.0, 0.1)`, `Th = 1 ms` (echo-state,
from the calibration rule on a fixed tiny set) — NO operating-point
search at this stage. Input `u_k ~ U{-1,+1}` (Bernoulli +/-1) for
BOTH tasks so a single shared reservoir drive feeds identical
features to both readouts. Pass = task scores stable to F3 AND no
pathological numerical failure. At this stage DO NOT interpret which
arm scores higher — well-posedness only.

## F5. Readout regularization frozen
Recall: ridge, fixed `lambda = 1e-2`, features standardized by
train mean/std, intercept fitted. Parity: logistic regression, L2,
`C = 1.0`, `class_weight='balanced'`, lbfgs, `tol=1e-4`,
`max_iter=1000`, decision threshold 0.5, same standardization.
Identical for every arm.

## F6. Topology seed = statistical unit
The 2 input sequences are within-seed repeats, NOT independent n.
Convergence criterion must hold on the PAIRED topology-level mean
Delta, and EVERY seed is inspected — a single catastrophic M13 seed
is scientifically different from a tiny median change hidden by
averaging; both are reported.

## Decision
Pass -> freeze harness -> full calibration -> locked test ladder.
M13 fails task-score convergence too -> STOP P2, do not rescue;
P3 conclusion strengthens to: low-bandwidth downstream probes are
not neutral enough to compare these event-based stiff spikers under
a common discretized interface. The deeper prize if it PASSES:
macroscopic computation stable despite discretization-sensitive
microscopic event placement is itself a hardware-relevant property
(silicon has jitter/mismatch/finite timing resolution too).

---
# P2 QUALIFICATION OUTCOME (2026-09-03): FAILED. Per-seed
# convergence criterion (F3+F6) not met by M13-kc1 (seed breach at
# d=8, boundary at d=2, systematic long-delay drift, near-null
# skill at the representative point). Positive result retained:
# task decoding attenuates event-timing sensitivity by 1-2 orders
# vs P1 features, but does NOT certify stable useful computation.
# Next: either preregistered P2b (min-skill-gated convergence, the
# LAST M13.5 attempt) or close M13.5 and move to M14.
