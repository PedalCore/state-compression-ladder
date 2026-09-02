"""M13 — velocity-collision diagnostic: is the compressed latent
a SUFFICIENT STATE?

For a trained encoder E, the future is recoverable from (z, I)
only if states colliding in latent space demand the same latent
velocity. Measure: sample states, compute z = E(x) and the
required velocity J_E(x) F_HH(x, I); within input-current bins,
find latent near-pairs (KD-tree) and report the dispersion of
their required velocities, normalized by typical velocity scale.
Calibration: dispersion of RANDOM pairs in the same bin.

High near-pair dispersion => no deterministic latent field of any
quality can serve this encoder — a lower bound on the k-ladder
independent of how G was trained.

python3 -m whitebox.hh_collision [--ckpts geo_k4_s0,geo_k4_s1]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_diag import hh_rhs, norm_state                   # noqa
from hh_geo import IS, Geo                               # noqa

OUT = pathlib.Path('results')
M = 200_000
NBINS_I = 8
RADIUS_FRAC = 0.05      # near-pair radius as fraction of z scale


def diagnose(ckpt, d, Str, Iflat):
    k = int(ckpt.split('_k')[1].split('_')[0])
    model = Geo(k)
    model.load_state_dict(torch.load(OUT / f'{ckpt}.pt',
                                     weights_only=True))
    model.eval()
    rng = np.random.default_rng(0)
    idx = rng.choice(Str.shape[0] * Str.shape[1], M, replace=False)
    X = torch.tensor(Str.reshape(-1, 4)[idx])
    Iv = Iflat[idx]
    F = torch.tensor(
        hh_rhs(X.numpy(), Iv), dtype=torch.float32)
    with torch.no_grad():
        pass
    z, zdot = jvp(model.E, (X,), (F,))
    z = z.detach().numpy()
    zdot = zdot.detach().numpy()
    vscale = float(np.median(np.linalg.norm(zdot, axis=1)) + 1e-9)
    from scipy.spatial import cKDTree
    qs = np.quantile(Iv, np.linspace(0, 1, NBINS_I + 1))
    near_d, rand_d = [], []
    for b in range(NBINS_I):
        m = (Iv >= qs[b]) & (Iv <= qs[b + 1])
        if m.sum() < 1000:
            continue
        zb, vb = z[m], zdot[m]
        r = RADIUS_FRAC * float(np.mean(zb.std(0)) + 1e-9) \
            * np.sqrt(zb.shape[1])
        tree = cKDTree(zb)
        pairs = tree.query_pairs(r, output_type='ndarray')
        if len(pairs) > 20000:
            pairs = pairs[np.random.default_rng(0).choice(
                len(pairs), 20000, replace=False)]
        if len(pairs) == 0:
            continue
        near_d.append(np.linalg.norm(
            vb[pairs[:, 0]] - vb[pairs[:, 1]], axis=1) / vscale)
        rr = np.random.default_rng(b)
        ra = rr.integers(0, len(vb), len(pairs))
        rb_ = rr.integers(0, len(vb), len(pairs))
        rand_d.append(np.linalg.norm(vb[ra] - vb[rb_], axis=1)
                      / vscale)
    near = np.concatenate(near_d) if near_d else np.array([np.nan])
    rand = np.concatenate(rand_d) if rand_d else np.array([np.nan])
    res = dict(ckpt=ckpt, k=k, pairs=int(len(near)),
               near_med=round(float(np.median(near)), 4),
               near_p95=round(float(np.quantile(near, 0.95)), 4),
               rand_med=round(float(np.median(rand)), 4),
               ratio_med=round(float(np.median(near)
                                     / (np.median(rand) + 1e-12)),
                               4))
    print('RESULT', json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', default='')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Iflat = d['train_I'].reshape(-1)
    ckpts = ([c for c in args.ckpts.split(',') if c] or
             sorted(p.stem for p in OUT.glob('geo_k*_s*.pt')))
    results = [diagnose(c, d, Str, Iflat) for c in ckpts]
    json.dump(results, open(OUT / 'collision_results.json', 'w'),
              indent=1)


if __name__ == '__main__':
    main()
