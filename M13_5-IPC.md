# M13.5 — Information Processing Capacity per hardware cost

Question (the programme's founding hypothesis, made quantitative):
does richer neuron dynamics buy computation per unit hardware?
Metric: Dambre Information Processing Capacity (linear memory +
nonlinear orders 2/3/4) + NARMA, DIVIDED BY state bits / MACs /
estimated gates. All neurons FROZEN (no primitive retraining —
M13 closed); only a linear readout trains. One shared reservoir
topology, matched conditions.

Arms (equal reservoir size, equal input drive, linear readout):
  LIF-1        (1 state, ~handful params)
  AdEx-2       (2 states)
  M13-field    (the frozen A0 vector field, 4-state)
  M13-kc1      (frozen field + 1-scalar trust corrector)
Timescale-spectrum control (the deep question — is it state COUNT
or the SPECTRUM those states carry?): a bank of leaky traces at
1 / 2 / 4 exponential timescales matched on state count.

Preregistered (before running):
P1: nonlinear IPC (orders 2-4) per state-bit is HIGHER for the
    HH-derived cells than LIF at matched cost.
P2: the timescale-bank control captures much of any advantage ->
    the resource is the timescale SPECTRUM, not HH geometry per se
    (the honest deflation to watch for).
P3: NARMA tracks IPC ordering.
Standing scope: measurement only; no primitive retraining; M14
(four-neuron re-taskability, synchronisation) waits on this
showing a per-cost advantage exists.

## v1 harness FLAWED (2026-09-03) — measured isolated single cells

v1 drove one cell of each type with the raw input and measured
IPC over its k states. Result: near-zero nonlinear capacity for
ALL cells (incl. nonlinear hh2), NARMA NaN (recursion diverged).
Not a finding — a design error: IPC is a RESERVOIR measurement;
a single deterministic unit has no population heterogeneity for a
linear readout to exploit, so nonlinear terms vanish. v1 numbers
DISCARDED. Fix (v2): reservoir of N heterogeneous cells per type
(random input weights, random biases/thresholds/timescale
jitter), IPC per TOTAL state count; NARMA input rescaled +
clipped to prevent divergence. This is the standard Dambre setup
and the honest apples-to-apples "computation per hardware" test.

## v2 SECOND confound (2026-09-03) — input-encoding mismatch

v2 fixed NARMA (no NaN; ~0.73 all cells) and reservoir structure,
but the trace banks alone got a tanh(win*x+bias) input front-end
while LIF/AdEx/hh2 got LINEAR drive. So traces' large nonlinear
IPC (C2/C3 ~0.8-1.85) measures the INPUT tanh, not dynamics; the
others show ~0 nonlinear capacity on linear drive. v2 cross-cell
numbers DISCARDED. Lesson: IPC comparison must fix the input
encoding identically across arms so nonlinear capacity is
attributable to DYNAMICS alone. This is a genuine measurement-
design problem (two confounds in two passes) — M13.5 is a real
project needing preregistered harness controls, not fast
iteration. PAUSING for harness design review rather than spawning
v3 blind — the same discipline that kept M13 solid.
Open design questions for v3: (a) identical linear input drive to
all cells (then nonlinear IPC can only come from dynamics —
expected: linear traces -> 0 nonlinear, by construction; spiking
reset + quadratic -> the real test); (b) hardware-cost proxy
beyond raw state count (spiking cells are cheaper per state than
continuous); (c) is near-zero nonlinear IPC for weakly-driven
LIF real or a drive-amplitude artifact.
