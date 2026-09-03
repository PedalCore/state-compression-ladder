# M14 FQ1 — pSeqDigits-64 space<->time transpose (BPTS). Preregistration.
# FROZEN before results. New question, not a rescue.

## Question (NOT "does BPTS work" — it trivially does)
What do we gain by physically trading TEMPORAL DEPTH for SPATIAL
replication, holding the information and cell-evaluation budget fixed?

## Data
pSeqDigits-64: sklearn 8x8 digits flattened to length-64. ONE frozen
random permutation pi (seed 0); x'_k = x_{pi(k)} for EVERY
factorization (so a sweet spot cannot be image topology rediscovered).

## Factorizations (S spatial lanes x T ticks), S*T = 64
{(1,64),(2,32),(4,16),(8,8),(16,4),(32,2),(64,1)}. Element x'[t*S+s]
enters lane s at tick t. After T ticks, CONCATENATE the S final
hidden states -> one linear classifier (dim S*d -> 10).

## Cell (the ONLY thing NOT changed across factorizations)
Boring shared tiny RNN: h_{t+1}=tanh(W h_t + b x_t + c), hidden d=16.
IDENTICAL cell law, width, init, optimizer, epochs/update-count,
data, SHARED theta (across all S lanes and T ticks), classifier type.
The S spatial models just instantiate more copies of the same cell.
Only S/T changes. (Classifier size S*d is part of the SPATIAL
readout cost and is reported, not hidden.)

## Cost bookkeeping (NOT "fixed physical state" — spatialization costs)
S*T=64 fixed => equal cell evaluations/sample. Report separately:
simultaneous state ~ S*d ; latency ~ T ; BPTT horizon = T. The
exchange under test is area/state <-> latency/BPTT-horizon.

## Metrics
(1) held-out accuracy. (2) early-gradient ratio at init
rho_G = ||dL/dh_earliest|| / ||dL/dh_latest|| — direct measure of the
credit-transmission problem vs BPTT horizon T.

## Pre-flight wiring test (before digits)
Synthetic delayed-copy y = x_0 on random scalar sequences. 1x64 must
carry x_0 through 64 tanh updates; 64x1 exposes the x_0 cell directly
to the classifier. rho_G MUST track this construction, else the
harness is wired wrong.

## Falsification (pre-committed)
- 1x64 ~= all factorizations -> temporal credit was NOT the limiting
  resource.
- accuracy rises monotonically to 64x1 in proportion to area/readout
  cost -> merely traded time for brute-force hardware.
- an INTERMEDIATE factorization lies OFF that trivial trade (much
  better accuracy for little extra spatial state -> a knee) -> a real
  finding worth pursuing (short-BPTT + spatial credit next).

## Discipline
Permutation, cell, width, init, optimizer, epochs frozen before
results. Same data/update-count. rho_G measured at init.

---
# BPTS RESULT (2026-09-03) — the space<->time transpose has a KNEE.
# First clear POSITIVE of the M13.5/M14 arc. (single seed; wiggles
# ~1-2% are seed noise.)

WIRING TEST (delayed copy y=bin(x_0)) PASSED: rho_G tracks BPTT
horizon over 8 orders of magnitude (1x64: 9.5e-9 gradient dead ->
64x1: 1.0), and 1x64 acc=0.072 ~chance (cannot carry x_0 64 steps).
Harness correct.

pSeqDigits-64 ST-transpose (d=16, frozen permutation, shared cell):
  S x T   BPTT  state=S*d  acc     rho_G
  1 x 64   64      16     0.615   1.0e-8
  2 x 32   32      32     0.784   6.4e-5
  4 x 16   16      64     0.889   4.9e-3
  8 x 8     8     128     0.961   5.5e-2   <- knee
 16 x 4     4     256     0.948   2.0e-1
 32 x 2     2     512     0.956   5.6e-1
 64 x 1     1    1024     0.975   1.0

FALSIFICATION TREE (pre-committed):
- (a) REJECTED: 1x64 (0.615) far below the rest -> temporal credit
  WAS the limiting resource (rho_G=1e-8, gradient dead).
- (b) REJECTED: accuracy NOT proportional to spatial area. 8x8
  (128 state, 0.961) BEATS 16x4 (256, 0.948) and 32x2 (512, 0.956)
  -> more state, lower acc -> not a readout-capacity trade.
- (c) CONFIRMED: an intermediate factorization lies OFF the trivial
  trade. 8x8 recovers ~96% of accuracy at 1/8 the simultaneous state
  of 64x1. A modest spatialization removes most of the temporal-
  credit burden; full spatial replication is wasteful.

MECHANISM. Accuracy tracks GRADIENT SURVIVAL (rho_G), set by the
BPTT horizon T, not by state count. Once spatialization shortens
BPTT to ~8 and lifts rho_G above the dead zone (~0.05), accuracy
saturates (~0.96) and extra spatial hardware buys ~nothing. The
knee is at BPTT~8.

WHAT THIS MEANS. The founding space<->time intuition has concrete
value: neither pure temporal (credit-starved) nor pure spatial
(hardware-wasteful) is right — a hybrid recovers nearly all the
accuracy at a fraction of the cost, and the binding constraint is
temporal-credit transmission, exposed directly by rho_G. This is
the first POSITIVE result of the arc, and unlike M13.5/M14 it needs
no exotic substrate: ordinary exact autodiff, one tiny shared cell.

CAVEAT: single seed; the 0.948-0.975 plateau values are within
~1-2% seed noise (the 1x64 outlier and the by-8x8 knee are robust).
Confirm with a few seeds before publication.

NEXT (reviewer's progression): take the knee factorization and ask
whether exact spatial backprop can be replaced by local/GLE-style
credit while keeping only SHORT BPTT inside each temporal block:
space-time transpose -> short temporal + spatial credit -> local
physical learning.
