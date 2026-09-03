# M14 — filed future questions (NOT rescues; open after write-up)

The relational-coordinate hypothesis on event-driven N-MNIST is
CLOSED (Exp-2/2b). Two sharp NEW questions are filed here, each a
fresh preregistration if ever taken up — not a continuation of the
falsified thesis.

## FQ1 — the space<->time transpose curve (BPTS)

BPTT is just reverse-mode differentiation through an unrolled causal
graph; "time" is only one axis to unroll. Backprop-through-space is
the same chain rule with index i instead of t (cf. Nature Comms
"Backpropagation through space, time and the brain"; GLE = an online
local approximation to spatio-temporal credit assignment). The
sharp, falsifiable M14 question is NOT "can we BPTS" (yes,
trivially) but:

  Can a temporal computation be TRANSPOSED into a spatial/relational
  representation so training no longer needs long-horizon BPTT — and
  what is the accuracy(N_space, T_time) curve at fixed effective
  state?

Experiment: same 64-step SeqDigits input, factor the sequence
  1x64, 2x32, 4x16, 8x8, 16x4, 32x2, 64x1
holding N_space * T_time = const. One extreme = pure temporal
recurrence / BPTT; other = pure spatial pipeline / plain backprop;
middle = hybrid. Match total state bits, ops, latency, throughput,
accuracy. This measures the engineering exchange
  temporal depth <-> spatial depth (area<->latency<->stored history)
and answers "how much computation should live in space vs time?" —
the programme's founding space<->time intuition as a curve. Delay-
line reservoirs (one cell + delay loop = many virtual spatial
nodes) and traveling-phase-wave arrays (time_past -> x_spatial) are
the two dual physical instantiations.

## FQ2 — Moving MNIST as the interface discriminator

CORRECTION to earlier disposition: Moving MNIST is NOT inherently
event-driven — it is FRAME-sequential with clean fixed update
boundaries, so the N-MNIST event-assignment pathology need NOT
occur. That makes it the natural discriminator between two
readings of the M13.5+M14 negatives:
  (a) temporal dynamical computation is intrinsically fragile, vs
  (b) EVENT-mediated interfaces are specifically where the fragility
      enters.
New question (fresh prereg, not a rescue): can dynamical state aid
PREDICTION when the temporal input interface itself is well-defined
(frame boundaries), where temporal evolution is still essential
(velocity) but event-assignment ambiguity is removed? A clean
future test; explicitly deferred.

## FQ3 — the un-run causal control (for honesty in the write-up)

We did NOT independently run "same event sequence, analytically
exact event integration vs timestep-assigned events". So the
input-perturbation mechanism is the BEST-SUPPORTED explanation of
the Exp-2/2b fragility, not proven causality. If the mechanism ever
needs to be nailed, that is the control to run.
