"""M13 control — classical 2-D HH reduction, NO training.

Krinsky-Kokoz/Rinzel-style: m = m_inf(V) (instantaneous), h = c - n
with c = mean(h + n) over the training data, leaving state (V, n).
Simulated under the teacher's exact protocol (0.01 ms internal,
recorded at 0.1 ms) and scored on the same metrics as every
learned arm. Purpose: a hand-designed 2-state sanity range for the
latent-field ladder's k=2 rung.

python3 -m whitebox.hh_reduced
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import (C, DT, ENA, EK, EL, GK, GL, GNA,  # noqa
                                 REC_EVERY, rates, spikes_from_v)
from hh_surrogate import spike_f1                         # noqa

OUT = pathlib.Path('results')


def simulate_2d(I_of_t, B, c):
    """Reduced model: state (V, n). I_of_t (B, N) at dt=DT."""
    V = np.full(B, -65.0)
    am, bm, ah, bh, an, bn = rates(V)
    n = an / (an + bn)
    N = I_of_t.shape[1]
    rec = np.empty((B, N // REC_EVERY), np.float32)
    for t in range(N):
        am, bm, ah, bh, an, bn = rates(V)
        tau_n = 1.0 / (an + bn)
        n_inf = an * tau_n
        n = n_inf + (n - n_inf) * np.exp(-DT / tau_n)
        m = am / (am + bm)                     # instantaneous
        h = np.clip(c - n, 0.0, 1.0)           # slaved
        ina = GNA * m ** 3 * h * (V - ENA)
        ik = GK * n ** 4 * (V - EK)
        il = GL * (V - EL)
        V = V + DT * (I_of_t[:, t] - ina - ik - il) / C
        V = np.clip(V, -120.0, 60.0)
        if t % REC_EVERY == REC_EVERY - 1:
            rec[:, t // REC_EVERY] = V
    return rec


def upsample(I_rec):
    """Recorded 0.1 ms drive -> 0.01 ms internal (hold)."""
    return np.repeat(I_rec, REC_EVERY, axis=1)


def main():
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    G = d['train_G']
    c = float((G[1] + G[2]).mean())
    print(f'slaving constant c = mean(h+n) = {c:.4f} '
          f'(std {(G[1] + G[2]).std():.4f})', flush=True)
    v_pred = simulate_2d(upsample(d['test_I']), len(d['test_I']), c)
    v_true = d['test_V']
    v_rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    f1 = spike_f1(v_true, v_pred)
    amps = d['fi_amps']
    N = int(1200.0 / DT)
    v_fi = simulate_2d(np.repeat(amps[:, None], N, 1), len(amps), c)
    rate = np.array([len(spikes_from_v(x[2000:])) for x in v_fi])
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    N2 = int(400.0 / DT)
    I2 = np.zeros((1, N2))
    I2[0, :N2 // 2] = -3.0
    v_r = simulate_2d(I2, 1, c)[0]
    reb = len(spikes_from_v(v_r[len(v_r) // 2:]))
    res = dict(arm='reduced-2d', c=round(c, 4),
               v_rmse_mv=round(v_rmse, 2), spike_f1=round(f1, 3),
               fi_rmse_hz=round(fi_rmse, 1), rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    json.dump(res, open(OUT / 'reduced2d_result.json', 'w'))


if __name__ == '__main__':
    main()
