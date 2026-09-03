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
