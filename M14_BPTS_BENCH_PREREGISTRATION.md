# M14 — BPTT vs BPTS engineering benchmark. Preregistration (frozen).
# First cheap test on pSeqDigits-64; scale to sMNIST-784 only if it wins.

## Architecture under test: parallel short temporal blocks + spatial
## reduction tree
Split the permuted 64-step sequence into S ordered chunks of length
T (S*T=64). Each chunk -> a SHARED recurrent cell run T steps (all S
blocks conceptually parallel, BPTT horizon = T). Combine the S block
states with an ORDER-SENSITIVE binary tree using a TINY SHARED
combiner G_phi(z_left, z_right) at every node -> root -> linear head.
Critical training depth ~ T + ceil(log2 S) vs full BPTT's S*T.

## Four arms
  full_bptt      1x64  Tdepth 64, no spatial credit   (conventional baseline)
  transpose+lin  8x8   Tdepth 8, concat+linear readout (existing positive)
  bpts_tree      8x8   Tdepth 8 + exact BP through 3 tree levels (ACTUAL BPTS)
  spatial_tree   64x1  Tdepth 1 + exact BP through 6 tree levels (spatial extreme)
Key contrast: bpts_tree vs transpose+lin — if ~equal, spatial credit
wasn't needed; if bpts_tree > transpose, the spatial hierarchy adds
something.

## Fairness
G is ONE tiny learned law shared across all tree nodes (not a dense
layer per node) — hardware-interesting and param-cheap. Report param
count, cell/G evals, simultaneous state, and critical-path depth
transparently for every arm (no hidden capacity).

## Metrics (CPU env: wall-clock reflects TOTAL compute, not critical
## path; the depth reduction is the THEORETICAL/hardware claim)
Per arm: test accuracy; forward/backward/step time (warmup+median);
peak activation memory (counted); params; cell+G evals; simultaneous
state; critical-path depth D. And the CPU-measurable optimization
signal: STEPS/epochs to reach 90% and 95% test accuracy (shorter
credit path should converge in fewer updates even if per-step cost is
similar).

## Pre-committed win levels
- NO WIN: BPTS ~ same accuracy but slower/more expensive, no faster
  convergence.
- OPTIMIZATION WIN: similar cost, reaches target accuracy in fewer
  updates (shorter surviving-gradient path).
- ARCHITECTURAL WIN: ~same accuracy as full BPTT AND materially
  better critical-path depth / memory (the wall-clock payoff needs
  parallel hardware to realize; report the depth + memory here).

## Scale-up (only if pSeqDigits shows optimization or architectural
## signal)
sMNIST-784 with a GRU cell (not vanilla tanh, which dies over long
horizons): factorizations 1x784 (full BPTT), 28x28, 49x16, 98x8.
D_BPTT=784 -> D_hybrid(49x16)=16+ceil(log2 49)~22. Question: does
spatializing temporal depth preserve accuracy while reducing training
latency / gradient-path depth / memory?

## Discipline
Frozen permutation, shared cell + shared G, same optimizer/epochs/
data; report both accuracy AND full cost ledger. If pSeqDigits shows
no optimization or architectural advantage -> STOP.
