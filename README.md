# The state-compression ladder

How many dynamical degrees of freedom does a neuron's computation
need — and which behaviours become representable, and in what
order, as state and precision become available?

Companion repository for the hi-sci-collab project
("The state-compression ladder", user PedalCore). The full lab
notebook — preregistrations, failed recipes, corrections, and
results, in chronological order — is `NOTEBOOK.md`. Failures are
first-class here: three training recipes failed their instrument
gate before the diagnostic fork found the cause, and the notebook
records all of it.

## Layout

- `hh_teacher.py` — Hodgkin-Huxley teacher (4 explicit states,
  exponential-Euler at 0.01 ms), OU + step drives, full-state
  recordings, OOD signatures (type II f-I curve, anodal-break
  rebound). Regenerates `results/hh_data_full.npz` (~26 MB, not
  committed): `python3 hh_teacher.py`
- `hh_surrogate.py` — observable-track ladder: k-state GRU
  surrogates from current (+ optional voltage feedback),
  free-run evaluation, instrument gate.
- `hh_classical.py` — hand-designed arms (LIF, Izhikevich, AdEx)
  fitted by CMA-ES to the same data; spikes are their explicit
  reset events.
- `hh_mech.py` — mechanistic-track ladder: full-state supervision
  through a k-latent bottleneck (encoder -> GRU cell -> decoder).
- `hh_diag.py` — the diagnostic fork (no latents, no recurrence):
  one-step flow map or analytic vector field + integrator,
  teacher-forced one-step error vs autonomous rollout, separated.

`results/` holds the JSON results and run logs for every
experiment in the notebook (checkpoints and the dataset are
regenerable, not committed).

## Headline state (2026-09-02, evolving)

- Learned vector field + integrator at ~1.5% relative error is
  the first learned model to fire the anodal-break rebound
  (exactly one, like the teacher), approximate the type II f-I
  curve (10.1 Hz RMSE), and reach spike F1 0.86 in free run —
  rollout fidelity improves CONTINUOUSLY with vector-field
  precision, and the signatures come online in a fixed order:
  firing -> f-I shape -> rebound.
- The same budget applied to discrete GRU transitions fails or
  underperforms drastically: how time and dynamics are
  represented matters more than latent state count so far.
- Hand-designed 1-2-state units (fitted by CMA-ES) beat every
  gradient-trained recurrent surrogate trained under matched
  early budgets — a trainability-vs-capacity distinction, not a
  capacity law.
- Multi-seed replication of apparent capacity boundaries is in
  progress; early seeds already show the k=3 -> 4 "transition"
  is substantially optimization probability.

Deps: `pip install numpy torch cma` (wandb optional).
