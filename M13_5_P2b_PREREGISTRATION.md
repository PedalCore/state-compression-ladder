# M13.5 P2b — convergence of USEFUL computation (final attempt)
# Preregistration. FROZEN before running. No rescue after this.

## Why P2b (and why it is new, not a repair)

P2's gate failed: M13-kc1 did not meet the per-seed convergence
criterion, AND the fixed representative operating point produced
near-null skill, so the test was underpowered. P2b asks the
strictly better question the P2 gate could not: **when the
primitive is actually performing useful computation, is that
computation stable to numerical refinement?** This is a NEW
preregistered experiment. It is the LAST M13.5 attempt; there is no
further rescue tree after it.

## The one legitimate design (no fishing possible)

1. **Equal-budget operating-point selection by VALIDATION SKILL
   only.** Same grid + same fixed search budget for every arm:
   `I0 in {0.0, 0.3}`, `g_in in {0.5, 1.0, 2.0}`,
   `g_rec in {0.0, 0.1, 0.3}` (18 configs/arm). Selection uses the
   COARSE step only (native n; M13 sub=10, trace/LIF dt=0.1).
   Selection objective = mean validation `S_recall` over d in
   {2, 8} on the calibration topologies, held-out sequence split.
   Selection NEVER looks at convergence — only at skill. Freeze
   `(I0,g_in,g_rec)*` per arm.
2. **Minimum-useful-skill gate (preregistered, absolute).** The
   frozen point must reach validation `R^2_recall(d=2) > 0.2` OR
   parity `acc(d=2) > 0.55` (materially above chance). If an arm
   reaches this at NO grid point, convergence is NOT assessed for
   it — an honest "cannot certify useful computation" outcome, not
   a pass and not a rescue.
3. **Convergence on UNTOUCHED data.** At the frozen point, run
   substeps n vs 2n on HELD-OUT topology seeds and fresh sequences
   (disjoint from calibration). Per-seed criterion (F3+F6):
   `|Delta R^2| < 0.03` (recall) and `|Delta acc| < 0.03` (parity)
   at EVERY seed and every assessed delay; report every seed;
   topology seed is the statistical unit.

## Frozen settings

- Calibration topology seeds: `{0, 1}`. Held-out test topology
  seeds: `{10, 11, 12, 13, 14, 15}` (6, disjoint). N = 30, Th = 1
  ms, wash = 200. Calibration seq: train 1500 / val 1000. Test
  seq: train 1500 / test 1000 (fresh draws, distinct RNG).
- Input `u_k ~ U{-1,+1}` shared across both tasks. Tasks + readouts
  exactly F2/F5 (2-bit parity `sign(u_{k-d} u_{k-2d})`; ridge
  lambda=1e-2 recall; balanced logistic parity). Skill metric F1.
- Arms: trace | LIF | M13-kc1 (the three qualification arms). No
  new arms until/unless P2b certifies the harness.

## Decision tree (pre-committed)

- **M13 reaches min-skill at some equal-budget point AND per-seed
  convergence holds (<0.03 all seeds/delays)** -> P2b PASS: M13
  supports a numerically stable USEFUL-computation benchmark ->
  proceed to the full per-cost ladder.
- **M13 reaches min-skill but per-seed convergence FAILS** ->
  useful decoded computation is NOT discretization-stable under the
  common protocol -> M13.5 closes with NO hardware-advantage claim;
  P2's negative becomes final.
- **M13 reaches min-skill at NO equal-budget point** -> cannot
  certify useful computation exists to test -> M13.5 closes with NO
  hardware-advantage claim.

In all three, the banked positive result stands: task decoding
attenuates event-timing sensitivity by 1-2 orders vs P1 features.
Any close-out moves to M14 (relational / synchronisation quantities
that are intrinsically less dependent on exact spike windows).

## Status
Frozen. Implementing and running now (the reviewer specified this
exact design). Result recorded whichever branch obtains.
