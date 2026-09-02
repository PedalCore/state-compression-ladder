"""M13 — collision/sufficiency diagnostic on FIXED (zero-learning)
representations: raw HH coordinates, classical projections, and
V alone. Same protocol as hh_collision/hh_delay: dispersion of the
hidden (m, h, n) among representation-space near-pairs at matched
input, normalized by random pairs.

raw4 (V,m,h,n) calibrates the protocol floor (legitimate within-
radius variation). v alone should reproduce D0's ~0.35.

python3 -m whitebox.hh_repr
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

REPRS = {
    'raw4': lambda V, G: np.stack(
        [V / 100.0, G[0], G[1], G[2]], 1),
    'vhn': lambda V, G: np.stack([V / 100.0, G[1], G[2]], 1),
    'vn': lambda V, G: np.stack([V / 100.0, G[2]], 1),
    'vh': lambda V, G: np.stack([V / 100.0, G[1]], 1),
    'v': lambda V, G: (V / 100.0)[:, None],
}


def run_repr(name, fn, V, G, I, rng):
    B, T = V.shape
    b_idx = rng.integers(0, B, M)
    t_idx = rng.integers(0, T, M)
    Vs = V[b_idx, t_idx]
    Gs = G[:, b_idx, t_idx]
    R = fn(Vs, Gs)
    H = Gs.T.copy()
    Iv = I[b_idx, t_idx]
    qs = np.quantile(Iv, np.linspace(0, 1, NBINS_I + 1))
    hscale = float(np.median(np.linalg.norm(
        H - H.mean(0), axis=1)) + 1e-9)
    near_d, rand_d = [], []
    for b in range(NBINS_I):
        m = (Iv >= qs[b]) & (Iv <= qs[b + 1])
        if m.sum() < 1000:
            continue
        Rb, Hb = R[m], H[m]
        r = RADIUS_FRAC * float(np.mean(Rb.std(0)) + 1e-9) \
            * np.sqrt(Rb.shape[1])
        tree = cKDTree(Rb)
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
    return dict(repr=name, dim=int(R.shape[1]),
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
    for name, fn in REPRS.items():
        res = run_repr(name, fn, V, G, I, rng)
        print('RESULT', json.dumps(res), flush=True)
        results.append(res)
    json.dump(results, open(OUT / 'repr_results.json', 'w'),
              indent=1)


if __name__ == '__main__':
    main()
