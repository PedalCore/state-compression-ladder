"""M13 — GEOMETRIC latent compression: tangent-space training,
no rollout, no BPTT.

z = E(x) (4 -> k), latent field G(z, I), decoder D (k -> 4).
Trained on independent state samples with three losses:
  recon        ||D(E(x)) - x||^2
  push-forward ||G(E(x), I) - J_E(x) F(x, I)||^2   (forward JVP)
  pull-back    ||J_D(z) G(z, I) - F(D(z), I)||^2
where F is the ANALYTIC HH vector field at the sample (normalized
units, per ms). Spike-region samples weighted 10x. Integration
(Euler substeps in z, decode each record step) happens only at
evaluation.

python3 -m whitebox.hh_geo [--ks 4,8,3,2,1] [--seeds 0,1]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import (DT, REC_EVERY, init_state,        # noqa
                                 spikes_from_v)
from hh_diag import hh_rhs, norm_state                    # noqa
from hh_surrogate import spike_f1                         # noqa

OUT = pathlib.Path('results')
VS, VOFF = 100.0, 65.0
IS = 10.0
SUB = 10


def mlp(nin, width, nout):
    return nn.Sequential(nn.Linear(nin, width), nn.Tanh(),
                         nn.Linear(width, width), nn.Tanh(),
                         nn.Linear(width, nout))


class Geo(nn.Module):
    def __init__(self, k, we=64, wg=128):
        super().__init__()
        self.E = mlp(4, we, k)
        self.G = mlp(k + 1, wg, k)
        self.D = mlp(k, we, 4)
        self.k = k

    def field(self, z, i):
        return self.G(torch.cat([z, i], -1))


def rollout(model, I_mv, x0, dev, bs=32):
    """Integrate z from E(x0), decode V each record step."""
    model.eval()
    outs = []
    dt_sub = DT * REC_EVERY / SUB
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            z = model.E(torch.tensor(
                np.repeat(x0[None], len(Ib), 0),
                dtype=torch.float32, device=dev))
            vs = []
            for t in range(Ib.shape[1]):
                i_t = Ib[:, t:t + 1]
                for _ in range(SUB):
                    z = z + dt_sub * model.field(z, i_t)
                z = torch.clamp(z, -30.0, 30.0)
                vs.append(model.D(z)[:, 0])
            outs.append(torch.stack(vs, 1).cpu().numpy()
                        * VS - VOFF)
    return np.concatenate(outs, 0)


def eval_all(model, d, Ste, rest, dev):
    v_pred = rollout(model, d['test_I'], rest, dev)
    v_true = Ste[..., 0] * VS - VOFF
    v_rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    f1 = spike_f1(v_true, v_pred)
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    v_fi = rollout(model, np.repeat(amps[:, None], T, 1), rest,
                   dev)
    rate = np.array([len(spikes_from_v(x[2000:])) for x in v_fi])
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    v_r = rollout(model, I2, rest, dev)[0]
    reb = len(spikes_from_v(v_r[T2 // 2:]))
    return v_rmse, f1, fi_rmse, reb


def train_one(k, seed, X, Inow, Fx, W, d, Ste, rest, dev, epochs):
    torch.manual_seed(seed)
    model = Geo(k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-4)
    nprm = sum(p.numel() for p in model.parameters())
    N = len(X)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = cnt = 0.0
        for b0 in range(0, N, 4096):
            idx = perm[b0:b0 + 4096]
            x = X[idx].to(dev)
            i = Inow[idx].to(dev)
            fx = Fx[idx].to(dev)
            w = W[idx].to(dev)
            z, jef = jvp(model.E, (x,), (fx,))       # E(x), J_E F
            g = model.field(z, i)
            xh, jdg = jvp(model.D, (z,), (g,))       # D(z), J_D G
            l_rec = ((xh - x) ** 2).mean(-1)
            l_push = ((g - jef) ** 2).mean(-1)
            l_pull = ((jdg - fx) ** 2).mean(-1)
            loss = ((l_rec + l_push + 0.5 * l_pull) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f'k={k} s={seed} ep{ep + 1}: geo loss '
                  f'{tot / cnt:.6f}', flush=True)
    train_s = time.time() - t0
    vr, f1, fi, reb = eval_all(model, d, Ste, rest, dev)
    res = dict(k=k, seed=seed, params=nprm,
               train_seconds=round(train_s, 1),
               v_rmse_mv=round(vr, 2), spike_f1=round(f1, 3),
               fi_rmse_hz=round(fi, 1), rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    torch.save(model.state_dict(), OUT / f'geo_k{k}_s{seed}.pt')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', default='4,8,3,2,1')  # control first
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    X = torch.tensor(Str.reshape(-1, 4))
    Inow = torch.tensor(d['train_I'].reshape(-1, 1) / IS,
                        dtype=torch.float32)
    Fx = torch.tensor(hh_rhs(Str, d['train_I']).reshape(-1, 4),
                      dtype=torch.float32)
    W = 1.0 + 9.0 * (X[:, 0] > 0.45).float()
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    results = [train_one(int(k), int(s), X, Inow, Fx, W, d, Ste,
                         rest, args.dev, args.epochs)
               for k in args.ks.split(',')
               for s in args.seeds.split(',')]
    json.dump(results, open(OUT / 'geo_results.json', 'w'),
              indent=1)
    print('=== GEO DONE ===', flush=True)


if __name__ == '__main__':
    main()
