"""M13 B2-delay — the field on DATA-NATIVE delay coordinates.

State = (V_t, V_{t-3ms}, ..., V_{t-12ms}) — the 5x3ms window that
D0 certified sufficient (ratio 0.140). Model: MLP predicting
dV/dt (analytic teacher supervision) from (delay window, I).
Training is iid/parallel (no BPTT, no autoencoder, no manufactured
off-manifold territory). Rollout: prime the buffer with 12 ms of
teacher voltage (or rest, for signature protocols), then free-run
with Euler substeps, re-querying with the updated V_t and held
lag components within each record step.

python3 -m whitebox.hh_b2 [--epochs 20] [--width 256]
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

from hh_teacher import DT, REC_EVERY, spikes_from_v      # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_surrogate import spike_f1                        # noqa

OUT = pathlib.Path('results')
IS = 10.0
VS, VOFF = 100.0, 65.0
LAGS = [0, 30, 60, 90, 120]        # record steps (0.1 ms each)
PRIME = 121
SUB = 10


def delay_matrix(Vn, t_idx, b_idx):
    """Vn (B,T) normalized voltage -> (M, len(LAGS))."""
    return np.stack([Vn[b_idx, t_idx - lg] for lg in LAGS], 1)


def mlp(width):
    return nn.Sequential(nn.Linear(len(LAGS) + 1, width),
                         nn.Tanh(), nn.Linear(width, width),
                         nn.Tanh(), nn.Linear(width, 1))


def rollout(model, I_mv, v_prime, dev, bs=32):
    """I_mv (B,T) raw current; v_prime (B,PRIME) normalized V to
    prime the buffer. Returns predicted V in mV (B,T)."""
    model.eval()
    outs = []
    dt_sub = DT * REC_EVERY / SUB
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            B, T = Ib.shape
            hist = torch.zeros(B, T, device=dev)
            hist[:, :PRIME] = torch.tensor(
                v_prime[b0:b0 + bs], dtype=torch.float32,
                device=dev)
            for t in range(PRIME, T):
                lagv = torch.stack(
                    [hist[:, t - 1 - lg] for lg in LAGS[1:]], 1)
                v = hist[:, t - 1]
                i_t = Ib[:, t - 1:t]
                for _ in range(SUB):
                    x = torch.cat([v[:, None], lagv, i_t], 1)
                    v = v + dt_sub * model(x).squeeze(-1)
                hist[:, t] = torch.clamp(v, -0.6, 1.6)
            outs.append(hist.cpu().numpy() * VS - VOFF)
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--denoise', type=float, default=0.0)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Vn = Str[..., 0]
    B, T = Vn.shape
    rng = np.random.default_rng(0)
    Midx = 2_000_000
    b_idx = rng.integers(0, B, Midx)
    t_idx = rng.integers(LAGS[-1], T, Midx)
    X = torch.tensor(delay_matrix(Vn, t_idx, b_idx),
                     dtype=torch.float32)
    Iv = d['train_I'][b_idx, t_idx]
    Inow = torch.tensor(Iv[:, None] / IS, dtype=torch.float32)
    dV = hh_rhs(Str[b_idx, t_idx], Iv)[:, 0]
    Y = torch.tensor(dV[:, None], dtype=torch.float32)
    scale = Y.std() + 1e-8
    W = 1.0 + 9.0 * (X[:, 0] > 0.45).float()
    torch.manual_seed(args.seed)
    model = mlp(args.width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4)
    N = len(X)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot = cnt = 0.0
        for b0 in range(0, N, 4096):
            idx = perm[b0:b0 + 4096]
            xw = X[idx]
            if args.denoise > 0:
                xw = xw.clone()
                xw[:, 1:] += args.denoise * torch.randn_like(
                    xw[:, 1:])       # corrupt lags, not V_t
            x = torch.cat([xw, Inow[idx]], 1).to(dev)
            y = Y[idx].to(dev)
            w = W[idx].to(dev)
            loss = ((((model(x) - y) / scale) ** 2).squeeze(-1)
                    * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f'b2 ep{ep + 1}: whitened MSE {tot / cnt:.6f}',
                  flush=True)
    ts = time.time() - t0
    # evaluation
    Ste = norm_state(d['test_V'], d['test_G'])
    Vte = Ste[..., 0]
    v_pred = rollout(model, d['test_I'], Vte[:, :PRIME], dev)
    v_true = d['test_V']
    sl = slice(PRIME, None)
    v_rmse = float(np.sqrt(np.mean(
        (v_pred[:, sl] - v_true[:, sl]) ** 2)))
    f1 = spike_f1(v_true[:, sl], v_pred[:, sl])
    rest_prime = np.full((1, PRIME), (-65.0 + VOFF) / VS)
    amps = d['fi_amps']
    Tfi = int(1200.0 / (DT * REC_EVERY))
    v_fi = rollout(model, np.repeat(amps[:, None], Tfi, 1),
                   np.repeat(rest_prime, len(amps), 0), dev)
    rate = np.array([len(spikes_from_v(x[2000:])) for x in v_fi])
    fi_rmse = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    v_r = rollout(model, I2, rest_prime, dev)[0]
    reb = len(spikes_from_v(v_r[T2 // 2:]))
    nprm = sum(p.numel() for p in model.parameters())
    arm = ('b2-delay-dn%.2f' % args.denoise
           if args.denoise > 0 else 'b2-delay')
    res = dict(arm=arm, seed=args.seed, params=nprm,
               train_seconds=round(ts, 1),
               v_rmse_mv=round(v_rmse, 2), spike_f1=round(f1, 3),
               fi_rmse_hz=round(fi_rmse, 1), rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'b2{"_dn" if args.denoise > 0 else ""}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
