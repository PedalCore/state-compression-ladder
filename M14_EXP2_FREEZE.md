# M14 Exp-2 (N-MNIST) — preprocessing FREEZE
# Frozen BEFORE any classification result is seen. The coordinate
# bridge: does information migrate from cell STATES into
# RELATIONSHIPS between cells when event timing is real?

## Acquisition is a DEPENDENCY, not part of the experiment

Tonic is used ONCE to download/parse N-MNIST, then each example is
written to a NEUTRAL cached format the experiment consumes:
per sample `(x, y, t, p, label)` raw arrays (int), plus a manifest
with dataset name, tonic version, per-split sample counts, and a
sha256 over the concatenated raw arrays. `m14_nmnist.py` reads ONLY
the cache — never tonic — so a future tonic transform/default
change cannot silently alter M14. Sensor is 34x34x2 (ON/OFF);
events returned in native `(x,y,t,p)` order.

## What is NOT done (would change the question)
- **No ToFrame / temporal binning** for the primary experiment —
  framing quantizes away the event timing we are testing. The only
  discretization permitted is whatever the dynamical integration
  itself requires (dt), and that is convergence-tested.
- **No saccade stabilization** (`stabilize=True` OFF). The camera
  motion IS the source of N-MNIST's temporal structure; removing it
  removes the phenomenon.
- **No polarity summing before the cells.** ON/OFF timing is
  genuine information.

## Frozen preprocessing pipeline
1. **Timestamp scaling:** subtract per-recording origin `t -= t.min()`
   (translation only); scale to the frozen substrate timescale by a
   single global constant `t_scale` (ms -> T0 units) chosen ONCE
   from the median recording duration, frozen. No per-sample
   rescaling.
2. **Receptive fields:** partition the 34x34 grid into `N` fixed
   non-overlapping tiles (N in {16, 32}; 16 = a 4x4 tiling of
   ~8x8 blocks). Tiling fixed before results; each event routes to
   its tile by (x,y). Frozen.
3. **Polarity-preserving drive:** each cell i receives
   `I_i(t) = g_plus * E_i^+(t) - g_minus * E_i^-(t)`, where
   `E_i^{+/-}(t)` are the ON/OFF event streams into tile i,
   delivered as instantaneous kicks at native event times (or a
   fixed narrow exponential kernel), affine and frozen. ON and OFF
   are NOT summed before the cell.
4. **Substrate:** the SAME lightweight phase system as Exp-1
   (uncoupled g_rec=0 and coupled g_rec>0), driven by `I_i(t)`.
   HH/M13 stay OUT (step 3). Same g_rec=0==uncoupled invariant,
   unit-tested first.

## Three coordinate VIEWS (identical trajectories; only the view differs)
- `R_state  = [cos(theta_i)]`            (absolute), dim N
- `R_events = [event rate / count per tile]` (absolute, event-domain)
- `R_rel    = [cos(theta_i-theta_j), sin(theta_i-theta_j)]` over
  exactly N/2 preselected pairs (fixed from the reservoir seed) so
  `dim R_rel = N`. A+R via the frozen PCA->N matched-dimension rule.
Kuramoto R_K diagnostic only, not in the primary classifier.

## Order of operations (reviewer-signed)
1. Freeze everything above (this doc).
2. dt-CONVERGENCE / representation gate FIRST: at the frozen
   receptive-field + polarity mapping, run integration dt vs dt/2
   on held-out samples; require per-sample-averaged classification
   score stability (|Delta acc| < 0.03) AND the g_rec=0 invariant,
   assessed only where acc is materially above chance (the SeqDigits
   metric refinement). Only if it passes:
3. CLASSIFY: 10-way logistic (F5 rules), matched-dimension views,
   report R_state vs R_events vs R_rel vs A+R, topology seed = unit,
   every seed reported.

## Pre-committed prediction (the weight of M14)
- sMNIST/SeqDigits: coupling may help; relational coordinates need
  NOT (artificial time). [SeqDigits-64: coupling helped, confirmed.]
- **N-MNIST: if M14 is right, `R_rel` becomes SPECIFICALLY useful —
  R_rel >= R_state at matched dimension — i.e. the information has
  migrated from cell states into relationships.** Falsified if
  R_rel <= R_state after matching, or if it fails the convergence
  gate.

## Status
Preprocessing frozen. Acquisition (tonic -> neutral cache) and the
convergence gate run before any classification number is read.
