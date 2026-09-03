"""M13 — the JEPA OBJECTIVE LADDER: same one-scalar SSM, same
field head; the arms differ only in the auxiliary objective that
shapes what the state means.

  none : dV regression only (baseline = joint-ssm1)
  raw  : + predict V_{t+5ms} from (window, c, I)
  multi: + predict V at +5/+10/+20 ms
  jepa : + predict the LATENT of the +10 ms future window
         (target = frozen random projection — collapse-proof;
         apparatus discarded at deployment)

All training parallel (scan + pointwise heads over teacher
windows). Rollout eval standard.

python3 -m whitebox.hh_jepa --obj jepa --seed 0
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_diag import hh_rhs, norm_state                   # noqa
from hh_surrogate import spike_f1                        # noqa
from hh_b2 import IS, LAGS, PRIME, VOFF, VS              # noqa
from hh_joint import JointModel, evaluate, rollout       # noqa

OUT = pathlib.Path('results')
NW = len(LAGS)
NF = NW + 1
TAUS = {'raw': [50], 'multi': [50, 100, 200], 'jepa': [100]}
ZDIM = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', default='jepa',
                    choices=['none', 'raw', 'multi', 'jepa'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    Vn_tr = Str[..., 0]
    B, T = Vn_tr.shape
    taus = TAUS.get(args.obj, [])
    tmax = max(taus) if taus else 0
    t_idx = np.arange(LAGS[-1], T - tmax - 1)
    Wd = np.stack([Vn_tr[:, t_idx - lg] for lg in LAGS], -1)
    Iw = d['train_I'][:, t_idx]
    X = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    Y = torch.tensor(hh_rhs(Str[:, t_idx], Iw)[..., 0],
                     dtype=torch.float32)
    scale = Y.std() + 1e-8
    Vw = torch.tensor(Vn_tr[:, t_idx], dtype=torch.float32)
    Wt = torch.where(Vw > 0.45, 10.0, 1.0)
    # future targets
    fut = {}
    for tau in taus:
        if args.obj == 'jepa':
            Wf = np.stack(
                [Vn_tr[:, t_idx + tau - lg] for lg in LAGS], -1)
            fut[tau] = torch.tensor(Wf, dtype=torch.float32)
        else:
            fut[tau] = torch.tensor(
                Vn_tr[:, t_idx + tau], dtype=torch.float32)
    torch.manual_seed(args.seed)
    model = JointModel(1, 'ssm').to(dev)
    aux = nn.ModuleDict()
    tgt_enc = None
    if args.obj == 'jepa':
        gen = torch.Generator().manual_seed(12345)
        tgt_enc = nn.Sequential(nn.Linear(NW, 16), nn.Tanh(),
                                nn.Linear(16, ZDIM))
        with torch.no_grad():
            for p_ in tgt_enc.parameters():
                p_.copy_(torch.randn(p_.shape, generator=gen)
                         * 0.5)
        for p_ in tgt_enc.parameters():
            p_.requires_grad_(False)
        tgt_enc = tgt_enc.to(dev)
        aux['p100'] = nn.Sequential(
            nn.Linear(NF + 1, 32), nn.Tanh(),
            nn.Linear(32, ZDIM))
    else:
        for tau in taus:
            aux[f'p{tau}'] = nn.Sequential(
                nn.Linear(NF + 1, 32), nn.Tanh(),
                nn.Linear(32, 1))
    aux = aux.to(dev)
    params = list(model.parameters()) + list(aux.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4)
    Vval = norm_state(d['val_V'], d['val_G'])[..., 0]
    t0 = time.time()
    best_vf1, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = torch.randperm(B)
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            x = X[idx].to(dev)
            y = Y[idx].to(dev)
            w_ = Wt[idx].to(dev)
            c = model.scan(x)
            pred = model.vdot(x, c)
            loss = ((((pred - y) / scale) ** 2) * w_).mean()
            xc = torch.cat([x, c], -1)
            for tau in taus:
                ft = fut[tau][idx].to(dev)
                if args.obj == 'jepa':
                    with torch.no_grad():
                        z_star = tgt_enc(ft)
                    z_hat = aux['p100'](xc)
                    loss = loss + ((z_hat - z_star) ** 2).mean()
                else:
                    v_hat = aux[f'p{tau}'](xc).squeeze(-1)
                    loss = loss + ((v_hat - ft) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        vp = rollout(model, d['val_I'], Vval[:, :PRIME],
                     dev).cpu().numpy() * VS - VOFF
        vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                             vp[:, PRIME:]))
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {k_: v_.clone() for k_, v_ in
                          model.state_dict().items()}
        print(f'jepa-{args.obj} s={args.seed} ep{ep + 1}: '
              f'val-F1 {vf1:.3f} (best {best_vf1:.3f})',
              flush=True)
    ts = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(model, d, Ste[..., 0], dev)
    res = dict(arm=f'obj-{args.obj}', seed=args.seed,
               train_seconds=round(ts, 1),
               best_val_f1=round(best_vf1, 3), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'obj_{args.obj}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
