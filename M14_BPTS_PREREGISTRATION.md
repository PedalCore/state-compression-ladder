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
