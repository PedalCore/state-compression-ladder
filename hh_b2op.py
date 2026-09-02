"""M13 B2-onpolicy — iterative corrective training on the model's
OWN malformed delay windows.

Round 0: clean iid tangent training (as B2 v1). Each round r:
roll out from teacher-primed buffers for a growing horizon,
collect the windows the model actually generates, pair each with
the clean teacher derivative at that time (corrective pairs),
retrain on clean + corrective data. Metrics per round: spike F1
and TIME-TO-DIVERGENCE (first crossing of |V_err| > 30 mV,
median over test sequences).

python3 -m whitebox.hh_b2op [--rounds 4] [--seed 0]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import DT, REC_EVERY, spikes_from_v      # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_surrogate import spike_f1                        # noqa
from hh_b2 import (IS, LAGS, PRIME, VOFF, VS,            # noqa
                            delay_matrix, mlp, rollout)

OUT = pathlib.Path('results')
HORIZONS = [50, 100, 200, 400]


def collect_corrective(model, Vn, I_raw, dV_true, horizon, dev,
                       nseq=64, rng=None):
    """Roll out from teacher-primed buffers over `horizon` steps;
    return (model windows, I, teacher dV targets)."""
    B, T = Vn.shape
    starts = rng.integers(PRIME, T - horizon - 1, nseq)
    seqs = rng.integers(0, B, nseq)
    dt_sub = DT * REC_EVERY / 10
    Xs, Is, Ys = [], [], []
    with torch.no_grad():
        hist = torch.zeros(nseq, PRIME + horizon)
        for j, (b, t0) in enumerate(zip(seqs, starts)):
            hist[j, :PRIME] = torch.tensor(
                Vn[b, t0 - PRIME:t0])
        Ib = torch.tensor(np.stack(
            [I_raw[b, t0 - PRIME:t0 + horizon]
             for b, t0 in zip(seqs, starts)]),
            dtype=torch.float32) / IS
        for t in range(PRIME, PRIME + horizon):
            lagv = torch.stack(
                [hist[:, t - 1 - lg] for lg in LAGS[1:]], 1)
            v = hist[:, t - 1]
            i_t = Ib[:, t - 1:t]
            for _ in range(10):
                x = torch.cat([v[:, None], lagv, i_t], 1)
                v = v + dt_sub * model(x.to(dev)).squeeze(-1).cpu()
            hist[:, t] = torch.clamp(v, -0.6, 1.6)
            # model's window at t, teacher target at (b, t0+t-PRIME)
            w = torch.cat([hist[:, t:t + 1], torch.stack(
                [hist[:, t - lg] for lg in LAGS[1:]], 1)], 1)
            Xs.append(w)
            Is.append(Ib[:, t:t + 1])
            Ys.append(torch.tensor(
                [dV_true[b, t0 + t - PRIME]
                 for b, t0 in zip(seqs, starts)],
                dtype=torch.float32)[:, None])
    return (torch.cat(Xs), torch.cat(Is), torch.cat(Ys))


def time_to_divergence(v_pred, v_true, thresh=30.0):
    """Median first-crossing time (ms) of |err| > thresh after
    PRIME; sequences that never diverge count as full length."""
    times = []
    for a, b in zip(v_pred, v_true):
        err = np.abs(a - b)[PRIME:]
        bad = np.flatnonzero(err > thresh)
        times.append((bad[0] if len(bad) else len(err)) * 0.1)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs0', type=int, default=12)
    ap.add_argument('--epochsr', type=int, default=6)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Vn = Str[..., 0]
    dV_true = hh_rhs(Str, d['train_I'])[..., 0]
    B, T = Vn.shape
    rng = np.random.default_rng(args.seed)
    Midx = 1_000_000
    b_idx = rng.integers(0, B, Midx)
    t_idx = rng.integers(LAGS[-1], T, Midx)
    Xc = torch.tensor(delay_matrix(Vn, t_idx, b_idx),
                      dtype=torch.float32)
    Ic = torch.tensor(
        d['train_I'][b_idx, t_idx][:, None] / IS,
        dtype=torch.float32)
    Yc = torch.tensor(dV_true[b_idx, t_idx][:, None],
                      dtype=torch.float32)
    scale = Yc.std() + 1e-8
    Wc = 1.0 + 9.0 * (Xc[:, 0] > 0.45).float()
    torch.manual_seed(args.seed)
    model = mlp(256).to(dev)
    Ste = norm_state(d['test_V'], d['test_G'])
    Vte = Ste[..., 0]

    def train(X, I, Y, W, epochs, lr):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        N = len(X)
        for ep in range(epochs):
            perm = torch.randperm(N)
            for b0 in range(0, N, 4096):
                idx = perm[b0:b0 + 4096]
                x = torch.cat([X[idx], I[idx]], 1).to(dev)
                loss = ((((model(x) - Y[idx].to(dev)) / scale)
                         ** 2).squeeze(-1)
                        * W[idx].to(dev)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def evaluate():
        v_pred = rollout(model, d['test_I'], Vte[:, :PRIME], dev)
        sl = slice(PRIME, None)
        f1 = spike_f1(d['test_V'][:, sl], v_pred[:, sl])
        ttd = time_to_divergence(v_pred, d['test_V'])
        vr = float(np.sqrt(np.mean(
            (v_pred[:, sl] - d['test_V'][:, sl]) ** 2)))
        return f1, ttd, vr

    t0 = time.time()
    train(Xc, Ic, Yc, Wc, args.epochs0, 1e-3)
    f1, ttd, vr = evaluate()
    log = [dict(round=0, horizon=0, f1=round(f1, 3),
                ttd_ms=round(ttd, 1), v_rmse=round(vr, 1))]
    print('ROUND', json.dumps(log[-1]), flush=True)
    Xo = Io = Yo = None
    for r in range(1, args.rounds + 1):
        H = HORIZONS[min(r - 1, len(HORIZONS) - 1)]
        xs, is_, ys = collect_corrective(
            model, Vn, d['train_I'], dV_true, H, dev, rng=rng)
        Xo = xs if Xo is None else torch.cat([Xo, xs])
        Io = is_ if Io is None else torch.cat([Io, is_])
        Yo = ys if Yo is None else torch.cat([Yo, ys])
        keep = rng.choice(len(Xc), 300_000, replace=False)
        Xr = torch.cat([Xc[keep], Xo])
        Ir = torch.cat([Ic[keep], Io])
        Yr = torch.cat([Yc[keep], Yo])
        Wr = torch.cat([Wc[keep],
                        3.0 * torch.ones(len(Xo))])
        train(Xr, Ir, Yr, Wr, args.epochsr, 3e-4)
        f1, ttd, vr = evaluate()
        log.append(dict(round=r, horizon=H,
                        corrective=int(len(Xo)),
                        f1=round(f1, 3), ttd_ms=round(ttd, 1),
                        v_rmse=round(vr, 1)))
        print('ROUND', json.dumps(log[-1]), flush=True)
    res = dict(arm='b2-onpolicy', seed=args.seed,
               train_seconds=round(time.time() - t0, 1),
               rounds=log)
    print('RESULT', json.dumps(res), flush=True)
    json.dump(res, open(OUT / f'b2op_s{args.seed}.json', 'w'))
    torch.save(model.state_dict(), OUT / f'b2op_s{args.seed}.pt')


if __name__ == '__main__':
    main()
