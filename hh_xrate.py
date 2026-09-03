"""M13 — THE EXCHANGE-RATE EXPERIMENT: how much contemporaneous
coupling is enough?

Mostly-parallel training (scan + vdot on teacher sequences) with
a fraction p of optimizer updates replaced by COUPLED updates:
a short (5 ms) rollout of the CURRENT model from teacher-primed
starts, BPTT through the model's own generated windows and state,
loss vs teacher V over the segment. Sweep p; the curve F1(p) is
the parallel-sequential exchange rate.

python3 -m whitebox.hh_xrate --p 0.05 --seed 0
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import DT, REC_EVERY                     # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_surrogate import spike_f1                        # noqa
from hh_b2 import IS, LAGS, PRIME, VOFF, VS              # noqa
from hh_joint import JointModel, evaluate, rollout       # noqa

OUT = pathlib.Path('results')
SUB = 10
SEG = 50                # coupled-segment length (5 ms)
NSEG = 32               # segments per coupled update


def coupled_update(model, opt, Vn, I_raw, dev, rng):
    """One BPTT update through a short CURRENT-model rollout."""
    B, T = Vn.shape
    starts = rng.integers(PRIME, T - SEG - 1, NSEG)
    seqs = rng.integers(0, B, NSEG)
    dt_sub = DT * REC_EVERY / SUB
    hist = torch.zeros(NSEG, PRIME + SEG)
    for j, (b, t0) in enumerate(zip(seqs, starts)):
        hist[j, :PRIME] = torch.tensor(Vn[b, t0 - PRIME:t0])
    Ib = torch.tensor(np.stack(
        [I_raw[b, t0 - PRIME:t0 + SEG]
         for b, t0 in zip(seqs, starts)]),
        dtype=torch.float32) / IS
    y = torch.tensor(np.stack(
        [Vn[b, t0:t0 + SEG]
         for b, t0 in zip(seqs, starts)]),
        dtype=torch.float32).to(dev)
    hist = hist.to(dev)
    Ib = Ib.to(dev)
    c = hist.new_zeros(NSEG, model.k)
    buf = [hist[:, i] for i in range(PRIME)]
    preds = []
    for t in range(SEG):
        ti = PRIME + t
        lagv = torch.stack(
            [buf[ti - 1 - lg] for lg in LAGS[1:]], 1)
        v = buf[ti - 1]
        i_t = Ib[:, ti - 1:ti]
        x = torch.cat([v[:, None], lagv, i_t], 1)
        c = model.step_c(x, c)
        for _ in range(SUB):
            xw = torch.cat([v[:, None], lagv, i_t], 1)
            v = v + dt_sub * model.vdot(xw, c)
        v = torch.clamp(v, -0.6, 1.6)
        buf.append(v)
        preds.append(v)
    pred = torch.stack(preds, 1)
    w_ = torch.where(y > 0.45, 10.0, 1.0)
    loss = (((pred - y) ** 2) * w_).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    Vn_tr = Str[..., 0]
    B, T = Vn_tr.shape
    t_idx = np.arange(LAGS[-1], T)
    Wd = np.stack([Vn_tr[:, t_idx - lg] for lg in LAGS], -1)
    Iw = d['train_I'][:, t_idx]
    X = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    Y = torch.tensor(hh_rhs(Str[:, t_idx], Iw)[..., 0],
                     dtype=torch.float32)
    scale = Y.std() + 1e-8
    Wt = torch.where(torch.tensor(
        Vn_tr[:, t_idx], dtype=torch.float32) > 0.45, 10.0, 1.0)
    Vval = norm_state(d['val_V'], d['val_G'])[..., 0]
    torch.manual_seed(args.seed)
    model = JointModel(1, 'ssm').to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    coupled_s = 0.0
    n_coupled = n_par = 0
    best_vf1, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = torch.randperm(B)
        for b0 in range(0, B, 32):
            if rng.random() < args.p:
                ts = time.time()
                coupled_update(model, opt, Vn_tr,
                               d['train_I'], dev, rng)
                coupled_s += time.time() - ts
                n_coupled += 1
            else:
                idx = perm[b0:b0 + 32]
                x = X[idx].to(dev)
                y = Y[idx].to(dev)
                w_ = Wt[idx].to(dev)
                c = model.scan(x)
                pred = model.vdot(x, c)
                loss = ((((pred - y) / scale) ** 2)
                        * w_).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                n_par += 1
        if (ep + 1) % 4 == 0:
            vp = rollout(model, d['val_I'], Vval[:, :PRIME],
                         dev).cpu().numpy() * VS - VOFF
            vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                                 vp[:, PRIME:]))
            if vf1 > best_vf1:
                best_vf1 = vf1
                best_state = {k_: v_.clone() for k_, v_ in
                              model.state_dict().items()}
            print(f'xr p={args.p} s={args.seed} ep{ep + 1}: '
                  f'val-F1 {vf1:.3f} (best {best_vf1:.3f}) '
                  f'coupled {n_coupled}', flush=True)
    ts_all = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(model, d, Ste[..., 0], dev)
    res = dict(arm=f'xrate-p{args.p}', seed=args.seed,
               total_seconds=round(ts_all, 1),
               coupled_seconds=round(coupled_s, 1),
               n_coupled=n_coupled, n_parallel=n_par,
               best_val_f1=round(best_vf1, 3), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'xrate_p{args.p}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
