# M14 — relational computation (opened 2026-09-03)

## Premise (handed over by M13.5)

Across the whole M13.5 arc the FRAGILE quantity was the absolute
event time t_i^spike; the STABLE, useful quantities are relational
and downstream. M14 tests whether useful computation lives in the
RELATIONSHIPS between cells (relative timing, phase, synchronisation,
cluster membership) rather than in richer individual cells — and, if
so, whether richer local dynamics help THERE where they did not help
as isolated reservoir primitives.

## v0 diagnostic (done) — relational observables ARE dt-stable

Small coupled HH net (N=8, heterogeneous supra-threshold drive,
one-scalar instantaneous coupling), run under dt and dt/2:
- ABSOLUTE spike timing drifts and ACCUMULATES: mean nearest-match
  shift 0.031 ms (first half) -> 0.102 ms (second half). The
  microscopic quantity is discretization-fragile, exactly as P1.
- RELATIONAL quantities are invariant under the same refinement:
  Kuramoto order parameter mean-diff 0.0005, R(t) correlation
  0.997; pairwise phase-difference cos error 0.0005. Partial-sync
  structure exists to measure (mean R = 0.25).
=> A global timing shift cancels in a difference; collective phase
structure survives numerical refinement that scrambles individual
spikes. The M14 premise holds. (m14_relational.py, m14_v0.json)

## Roadmap — the MNIST-temporal benchmark ladder (see M14_BENCHMARK.md)

Benchmark pivoted (reviewer, 2026-09-03) from synthetic recall/
parity to real temporal datasets that test increasingly meaningful
notions of time: sMNIST -> N-MNIST -> Moving MNIST =
memory -> relationships -> prediction. The synthetic-task step-2
draft (M14_STEP2_PREREGISTRATION.md) is SUPERSEDED before sign-off;
its coordinate-system methodology (A|R|A+R matched-dim views,
convergence-before-performance) is preserved and lands on N-MNIST.

1. STABILITY (done, v0): relational observables step-halving stable
   where t_spike is not. YES.
2. EXP-1 sMNIST (geometry/memory, OFFLINE-runnable via sklearn
   digits 8x8): cell ladder trace|LIF|oscillator|coupled at equal
   N + presentation-SPEED sweep T_pixel in {0.25..4} + convergence
   gate. Q: does matching temporal geometry matter more than raw
   state count? [prereg drafted, awaiting sign-off]
3. EXP-2 N-MNIST (the coordinate bridge; needs data + event loader,
   tonic not installed): compress event field to N receptive
   fields -> cells; compare R_state|R_events|R_rel from identical
   trajectories. Q: does relational timing expose the class more
   efficiently than absolute state? (invariance->decodability->
   utility on REAL event timing.) Pre-committed contrast: relational
   advantage expected here, NOT on sMNIST.
4. EXP-3 Moving MNIST (prediction): observe 5 frames, predict
   position 5 ahead; recurrence must represent velocity.
5. Only after the bridge holds: step-3 richness test (does richer
   local dynamics build a better relational basis at matched size)
   and the ONN framing (Todri-Sanial review in refs/).

## Standing discipline (unchanged)
Preregister before comparative runs; failures first-class; no
harness/operating-point tuning until a spiking arm 'looks good';
relational observables and costs defined identically across arms.
