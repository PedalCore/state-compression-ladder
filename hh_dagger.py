"""M13 — DAgger for dynamical state: closed-loop dataset
aggregation for the joint observable SSM.

Rounds: (1) parallel retrain (SSM scan on teacher sequences +
field regression on teacher points AND aggregated corrective
pairs); (2) short rollouts collect the model's actual
(window, c) visits, frontier-concentrated; (3) teacher-label
(V_dot at matched times); (4) aggregate, grow horizon.
Sequential cost = data collection only.

python3 -m whitebox.hh_dagger --seed 0
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
from hh_b2op import time_to_divergence                   # noqa

OUT = pathlib.Path('results')
SUB = 10
HORIZONS = [100, 200, 400, 800, 1500]      # record steps


def collect(model, Vn, I_raw, Ydot, horizon, dev, nseq, rng,
            align='phase'):
    """Short rollouts from teacher-primed buffers; collect the
    model's (window, c) at every step. align='clock': label with
    teacher V_dot at the same time index. align='phase': label
    with teacher V_dot at the NEAREST teacher window within
    +-30 steps (dynamical similarity, not clock). Returns
    (Xc [N, NF], Cc [N, k], Yc [N])."""
    B, T = Vn.shape
    starts = rng.integers(PRIME, T - horizon - 1, nseq)
    seqs = rng.integers(0, B, nseq)
    dt_sub = DT * REC_EVERY / SUB
    Xs, Cs, Ys = [], [], []
    with torch.no_grad():
        hist = torch.zeros(nseq, PRIME + horizon)
        for j, (b, t0) in enumerate(zip(seqs, starts)):
            hist[j, :PRIME] = torch.tensor(
                Vn[b, t0 - PRIME:t0])
        Ib = torch.tensor(np.stack(
            [I_raw[b, t0 - PRIME:t0 + horizon]
             for b, t0 in zip(seqs, starts)]),
            dtype=torch.float32) / IS
        c = hist.new_zeros(nseq, model.k)
        for t in range(PRIME, PRIME + horizon):
            lagv = torch.stack(
                [hist[:, t - 1 - lg] for lg in LAGS[1:]], 1)
            v = hist[:, t - 1]
            i_t = Ib[:, t - 1:t]
            x = torch.cat([v[:, None], lagv, i_t], 1)
            c = model.step_c(x.to(dev), c.to(dev)).cpu()
            vv = v.clone()
            for _ in range(SUB):
                xw = torch.cat([vv[:, None], lagv, i_t], 1)
                vv = vv + dt_sub * model.vdot(
                    xw.to(dev), c.to(dev)).cpu()
            hist[:, t] = torch.clamp(vv, -0.6, 1.6)
            Xs.append(x)
            Cs.append(c.clone())
            if align == 'clock':
                Ys.append(torch.tensor(
                    [Ydot[b, t0 + t - PRIME]
                     for b, t0 in zip(seqs, starts)],
                    dtype=torch.float32))
            else:
                labels = []
                wm = x[:, :len(LAGS)].numpy()
                for j, (b, t0) in enumerate(zip(seqs, starts)):
                    tc = t0 + t - PRIME
                    lo = max(tc - 30, LAGS[-1])
                    hi = min(tc + 30, Vn.shape[1] - 1)
                    cand = np.stack(
                        [Vn[b, np.arange(lo, hi) - lg]
                         for lg in LAGS], -1)
                    dists = np.linalg.norm(cand - wm[j], axis=1)
                    tstar = lo + int(np.argmin(dists))
                    labels.append(Ydot[b, tstar])
                Ys.append(torch.tensor(labels,
                                       dtype=torch.float32))
    return torch.cat(Xs), torch.cat(Cs), torch.cat(Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--align', default='phase',
                    choices=['phase', 'clock'])
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--epochs0', type=int, default=15)
    ap.add_argument('--epochsr', type=int, default=6)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    Vn_tr = Str[..., 0]
    B, T = Vn_tr.shape
    Ydot_full = hh_rhs(Str, d['train_I'])[..., 0]
    t_idx = np.arange(LAGS[-1], T)
    Wd = np.stack([Vn_tr[:, t_idx - lg] for lg in LAGS], -1)
    Iw = d['train_I'][:, t_idx]
    X = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    Y = torch.tensor(Ydot_full[:, t_idx], dtype=torch.float32)
    scale = Y.std() + 1e-8
    Vw = torch.tensor(Vn_tr[:, t_idx], dtype=torch.float32)
    Wt = torch.where(Vw > 0.45, 10.0, 1.0)
    Vval = norm_state(d['val_V'], d['val_G'])[..., 0]
    torch.manual_seed(args.seed)
    model = JointModel(1, 'ssm').to(dev)
    rng = np.random.default_rng(args.seed)
    Xc = Cc = Yc = None
    log = []
    best_vf1, best_state = -1.0, None

    def train(epochs, lr):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(B)
            for b0 in range(0, B, 32):
                idx = perm[b0:b0 + 32]
                x = X[idx].to(dev)
                y = Y[idx].to(dev)
                w_ = Wt[idx].to(dev)
                c = model.scan(x)
                pred = model.vdot(x, c)
                loss = ((((pred - y) / scale) ** 2) * w_).mean()
                if Xc is not None:
                    j = torch.randint(0, len(Xc), (8192,))
                    pc = model.vdot(Xc[j].to(dev),
                                    Cc[j].to(dev))
                    loss = loss + 0.5 * (((pc - Yc[j].to(dev))
                                          / scale) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    t0 = time.time()
    seq_seconds = 0.0
    for r in range(args.rounds + 1):
        if r == 0:
            train(args.epochs0, 1e-3)
        else:
            H = HORIZONS[min(r - 1, len(HORIZONS) - 1)]
            ts = time.time()
            xs, cs, ys = collect(model, Vn_tr, d['train_I'],
                                 Ydot_full, H, dev, 96, rng,
                                 align=args.align)
            seq_seconds += time.time() - ts
            Xc = xs if Xc is None else torch.cat([Xc, xs])
            Cc = cs if Cc is None else torch.cat([Cc, cs])
            Yc = ys if Yc is None else torch.cat([Yc, ys])
            train(args.epochsr, 3e-4)
        vp = rollout(model, d['val_I'], Vval[:, :PRIME],
                     dev).cpu().numpy() * VS - VOFF
        vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                             vp[:, PRIME:]))
        ttd = time_to_divergence(vp, d['val_V'])
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {k_: v_.clone() for k_, v_ in
                          model.state_dict().items()}
        log.append(dict(round=r, val_f1=round(vf1, 3),
                        ttd_ms=round(ttd, 1),
                        corrective=0 if Xc is None else len(Xc),
                        seq_s=round(seq_seconds, 1)))
        print('ROUND', json.dumps(log[-1]), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(model, d, Ste[..., 0], dev)
    res = dict(arm=f'dagger-ssm1-{args.align}',
               seed=args.seed,
               total_seconds=round(time.time() - t0, 1),
               seq_seconds=round(seq_seconds, 1),
               best_val_f1=round(best_vf1, 3), rounds=log, **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'dagger_{args.align}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
