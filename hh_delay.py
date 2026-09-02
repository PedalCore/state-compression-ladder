"""M13 D0 — delay-coordinate sufficiency, NO TRAINING.

Does history-as-geometry make the future single-valued from
voltage alone? For delay vectors (V_t, V_{t-tau}, ...), find
near-pairs at matched input current and measure the dispersion of
the hidden gate state (m, h, n) they imply, normalized by the
random-pair dispersion in the same input bin. Takens predicts the
ratio falls toward a floor as delays are added.

python3 -m whitebox.hh_delay
"""

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

OUT = pathlib.Path('results')
M = 200_000
NBINS_I = 8
RADIUS_FRAC = 0.05
CONFIGS = [(1, 10), (2, 10), (3, 10), (5, 10),
           (2, 30), (3, 30), (5, 30)]      # (n_delays, lag_steps)


def run_config(nd, lag, V, G, I, rng):
    B, T = V.shape
    maxlag = lag * (nd - 1)
    b_idx = rng.integers(0, B, M)
    t_idx = rng.integers(maxlag, T, M)
    D = np.stack([V[b_idx, t_idx - j * lag] for j in range(nd)],
                 axis=1) / 100.0            # scale ~ normalized V
    H = np.stack([G[0, b_idx, t_idx], G[1, b_idx, t_idx],
                  G[2, b_idx, t_idx]], axis=1)
    Iv = I[b_idx, t_idx]
    qs = np.quantile(Iv, np.linspace(0, 1, NBINS_I + 1))
    hscale = float(np.median(np.linalg.norm(
        H - H.mean(0), axis=1)) + 1e-9)
    near_d, rand_d = [], []
    for b in range(NBINS_I):
        m = (Iv >= qs[b]) & (Iv <= qs[b + 1])
        if m.sum() < 1000:
            continue
        Db, Hb = D[m], H[m]
        r = RADIUS_FRAC * float(np.mean(Db.std(0)) + 1e-9) \
            * np.sqrt(nd)
        tree = cKDTree(Db)
        pairs = tree.query_pairs(r, output_type='ndarray')
        if len(pairs) > 20000:
            pairs = pairs[rng.choice(len(pairs), 20000,
                                     replace=False)]
        if len(pairs) == 0:
            continue
        near_d.append(np.linalg.norm(
            Hb[pairs[:, 0]] - Hb[pairs[:, 1]], axis=1) / hscale)
        ra = rng.integers(0, len(Hb), len(pairs))
        rb = rng.integers(0, len(Hb), len(pairs))
        rand_d.append(np.linalg.norm(Hb[ra] - Hb[rb], axis=1)
                      / hscale)
    near = np.concatenate(near_d)
    rand = np.concatenate(rand_d)
    return dict(n_delays=nd, lag_ms=lag * 0.1,
                pairs=int(len(near)),
                near_med=round(float(np.median(near)), 4),
                near_p95=round(float(np.quantile(near, 0.95)), 4),
                ratio_med=round(float(np.median(near)
                                      / (np.median(rand) + 1e-12)),
                                4))


def main():
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    V, G, I = d['train_V'], d['train_G'], d['train_I']
    rng = np.random.default_rng(0)
    results = []
    for nd, lag in CONFIGS:
        res = run_config(nd, lag, V, G, I, rng)
        print('RESULT', json.dumps(res), flush=True)
        results.append(res)
    json.dump(results, open(OUT / 'delay_results.json', 'w'),
              indent=1)


if __name__ == '__main__':
    main()
