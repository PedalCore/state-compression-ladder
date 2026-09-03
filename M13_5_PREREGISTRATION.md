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
