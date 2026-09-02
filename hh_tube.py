"""M13 — the tube experiment: how little sequential supervision
turns a parallel local description of dynamics into a stable
global flow?

Arms (same Geo architecture, k=4, seeds {0,1}):
  geo_noise    tube points z' = z + eps with DECODE-GROUNDED
               targets: x' = D(z'), target = J_E(x') F_HH(x', I)
  geo_restore  tube + manifold attraction: target -= LAM *
               (z' - E(D(z')))
  geo_onpolicy each epoch, supervise the latent states the model
               ACTUALLY visits in short rollouts (drift-targeted)
  rollout      latent TBPTT reference (SUB=2 during training)

python3 -m whitebox.hh_tube --arm geo_noise --seed 0
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch.func import jvp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import DT, REC_EVERY, init_state         # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_geo import IS, VOFF, VS, Geo, eval_all           # noqa

OUT = pathlib.Path('results')
LAM = 1.0          # manifold-restoring rate, per ms
NOISE = 0.1        # tube noise, fraction of per-dim latent std
TUBE_W = 1.0       # tube-loss weight vs manifold losses


def rhs_torch(x_norm, i_raw):
    """Analytic HH field at decoded (normalized) states, torch."""
    f = hh_rhs(x_norm.detach().cpu().numpy(),
               i_raw.detach().cpu().numpy())
    return torch.tensor(f, dtype=torch.float32,
                        device=x_norm.device)


def tube_loss(model, z_pts, i_pts, restore=False):
    """Decode-grounded tangent supervision at off-manifold z."""
    x_p = model.D(z_pts).detach()
    x_p = torch.clamp(x_p, -0.2, 1.2)
    f_p = rhs_torch(x_p, i_pts.squeeze(-1) * IS)
    _, target = jvp(model.E, (x_p,), (f_p,))
    target = target.detach()
    if restore:
        proj = model.E(model.D(z_pts)).detach()
        target = target - LAM * (z_pts - proj)
    g = model.field(z_pts, i_pts)
    return ((g - target) ** 2).mean()


def short_rollout_states(model, X0, I_seq, dev, steps=20):
    """Collect latent states visited by the CURRENT model."""
    dt_sub = DT * REC_EVERY / 10
    with torch.no_grad():
        z = model.E(X0.to(dev))
        zs = []
        for t in range(steps):
            i_t = I_seq[:, t:t + 1].to(dev)
            for _ in range(10):
                z = z + dt_sub * model.field(z, i_t)
            z = torch.clamp(z, -30.0, 30.0)
            zs.append((z.clone(), i_t.clone()))
    return zs


def train_geo_variant(arm, seed, d, Str, Ste, rest, dev, epochs,
                      k=4):
    torch.manual_seed(seed)
    model = Geo(k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-4)
    X = torch.tensor(Str.reshape(-1, 4))
    Inow = torch.tensor(d['train_I'].reshape(-1, 1) / IS,
                        dtype=torch.float32)
    Fx = torch.tensor(hh_rhs(Str, d['train_I']).reshape(-1, 4),
                      dtype=torch.float32)
    W = 1.0 + 9.0 * (X[:, 0] > 0.45).float()
    Itr_seq = torch.tensor(d['train_I'] / IS,
                           dtype=torch.float32)
    Xtr_seq = torch.tensor(Str)
    N = len(X)
    t0 = time.time()
    onpol = []
    for ep in range(epochs):
        if arm == 'geo_onpolicy' and ep > 0:
            starts = torch.randint(0, len(Xtr_seq), (200,))
            t_off = int(torch.randint(0, Xtr_seq.shape[1] - 25,
                                      (1,)))
            onpol = short_rollout_states(
                model, Xtr_seq[starts, t_off],
                Itr_seq[starts, t_off:t_off + 20], dev)
        perm = torch.randperm(N)
        tot = cnt = 0.0
        for b0 in range(0, N, 4096):
            idx = perm[b0:b0 + 4096]
            x = X[idx].to(dev)
            i = Inow[idx].to(dev)
            fx = Fx[idx].to(dev)
            w = W[idx].to(dev)
            z, jef = jvp(model.E, (x,), (fx,))
            g = model.field(z, i)
            xh, jdg = jvp(model.D, (z,), (g,))
            l_rec = ((xh - x) ** 2).mean(-1)
            l_push = ((g - jef) ** 2).mean(-1)
            l_pull = ((jdg - fx) ** 2).mean(-1)
            loss = ((l_rec + l_push + 0.5 * l_pull) * w).mean()
            if arm in ('geo_noise', 'geo_restore'):
                zstd = z.detach().std(0, keepdim=True)
                zp = z.detach() + NOISE * zstd * torch.randn_like(
                    z)
                loss = loss + TUBE_W * tube_loss(
                    model, zp, i, restore=(arm == 'geo_restore'))
            elif arm == 'geo_onpolicy' and onpol:
                j = int(torch.randint(0, len(onpol), (1,)))
                zp, ip = onpol[j]
                loss = loss + TUBE_W * tube_loss(model, zp, ip)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f'{arm} s={seed} ep{ep + 1}: loss '
                  f'{tot / cnt:.6f}', flush=True)
    return model, time.time() - t0


def train_rollout(seed, d, Str, dev, epochs=6, chunk=250, sub=2,
                  k=4):
    torch.manual_seed(seed)
    model = Geo(k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    Ytr = torch.tensor(Str)
    Wt = torch.ones(Ytr.shape[:2])
    Wt[Ytr[..., 0] > 0.45] = 10.0
    dt_sub = DT * REC_EVERY / sub
    B, T = Itr.shape
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(B)
        tot = cnt = 0.0
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            z = None
            for c0 in range(0, T, chunk):
                x = Itr[idx, c0:c0 + chunk].to(dev)
                y = Ytr[idx, c0:c0 + chunk].to(dev)
                w = Wt[idx, c0:c0 + chunk].to(dev)
                z = (model.E(y[:, 0]) if z is None
                     else z.detach())
                preds = []
                for t in range(x.shape[1]):
                    i_t = x[:, t:t + 1]
                    for _ in range(sub):
                        z = z + dt_sub * model.field(z, i_t)
                    preds.append(model.D(z))
                pred = torch.stack(preds, 1)
                loss = (((pred - y) ** 2).mean(-1) * w).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                z = z.detach()
                tot += float(loss) * x.numel()
                cnt += x.numel()
        print(f'rollout s={seed} ep{ep + 1}: loss {tot / cnt:.5f}',
              flush=True)
    return model, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=['geo_noise', 'geo_restore',
                             'geo_onpolicy', 'rollout'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    if args.arm == 'rollout':
        model, ts = train_rollout(args.seed, d, Str, args.dev,
                                  k=args.k)
    else:
        model, ts = train_geo_variant(args.arm, args.seed, d, Str,
                                      Ste, rest, args.dev,
                                      args.epochs, k=args.k)
    vr, f1, fi, reb = eval_all(model, d, Ste, rest, args.dev)
    res = dict(arm=args.arm, k=args.k, seed=args.seed,
               train_seconds=round(ts, 1), v_rmse_mv=round(vr, 2),
               spike_f1=round(f1, 3), fi_rmse_hz=round(fi, 1),
               rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    json.dump(res, open(
        OUT / f'tube_{args.arm}_k{args.k}_s{args.seed}.json',
        'w'))


if __name__ == '__main__':
    main()
