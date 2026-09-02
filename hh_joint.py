"""M13 — JOINT observable model: selective-scan state participates
from the start.

  c_{t+1} = A(x_t) c_t + B(x_t)      (x_t = window + I; parallel
                                      scan under teacher windows)
  dV/dt   = F(window, c, I)          (trained on analytic targets
                                      over all timepoints at once)

kind='ssm': fully parallel joint training (scan + pointwise F).
kind='gru': same architecture with GRUCell state — TBPTT through
the c-chain (the sequential price the SSM claims to avoid).
Deployment: sequential rollout, self-generated window, carried c.

python3 -m whitebox.hh_joint --kind ssm --k 1 --seed 0
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
from hh_b2 import IS, LAGS, PRIME, VOFF, VS              # noqa
from hh_comp import f1_by_window                         # noqa

OUT = pathlib.Path('results')
SUB = 10
SCAN_CHUNK = 250
NW = len(LAGS)
NF = NW + 1


class JointModel(nn.Module):
    def __init__(self, k=1, kind='ssm', width=256):
        super().__init__()
        self.k = k
        self.kind = kind
        if kind == 'ssm':
            self.wa = nn.Linear(NF, k)
            self.wb = nn.Linear(NF, k)
        else:
            self.cell = nn.GRUCell(NF, k)
        self.F = nn.Sequential(
            nn.Linear(NF + k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1))

    def coeffs(self, X):
        tau = 1.0 + torch.nn.functional.softplus(self.wa(X))
        return -0.1 / tau, 0.1 * torch.tanh(self.wb(X))

    def scan(self, X, c0=None):
        """(B, T, NF) -> c (B, T, k), parallel closed form."""
        B, T, _ = X.shape
        log_a, b = self.coeffs(X)
        cs = []
        c_prev = (X.new_zeros(B, self.k) if c0 is None else c0)
        for s in range(0, T, SCAN_CHUNK):
            la = log_a[:, s:s + SCAN_CHUNK]
            bb = b[:, s:s + SCAN_CHUNK]
            G = torch.cumsum(la, 1)
            inner = torch.cumsum(torch.exp(-G) * bb, 1)
            c = torch.exp(G) * (c_prev.unsqueeze(1) + inner)
            cs.append(c)
            c_prev = c[:, -1]
        return torch.cat(cs, 1)

    def gru_pass(self, X, c0=None):
        B, T, _ = X.shape
        c = (X.new_zeros(B, self.k) if c0 is None else c0)
        cs = []
        for t in range(T):
            c = self.cell(X[:, t], c)
            cs.append(c)
        return torch.stack(cs, 1)

    def vdot(self, X, c):
        return self.F(torch.cat([X, c], -1)).squeeze(-1)

    def step_c(self, x, c):
        """Single-step state update for rollout. x (B, NF)."""
        if self.kind == 'ssm':
            la, b = self.coeffs(x)
            return torch.exp(la) * c + b
        return self.cell(x, c)


def rollout(model, I_mv, v_prime, dev, bs=32):
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
            c = hist.new_zeros(B, model.k)
            for t in range(PRIME, T):
                lagv = torch.stack(
                    [hist[:, t - 1 - lg] for lg in LAGS[1:]], 1)
                v = hist[:, t - 1]
                i_t = Ib[:, t - 1:t]
                x = torch.cat([v[:, None], lagv, i_t], 1)
                c = model.step_c(x, c)
                for _ in range(SUB):
                    xw = torch.cat([v[:, None], lagv, i_t], 1)
                    v = v + dt_sub * model.vdot(xw, c)
                hist[:, t] = torch.clamp(v, -0.6, 1.6)
            outs.append(hist)
    return torch.cat(outs, 0)


def evaluate(model, d, Vn_te, dev):
    vp = rollout(model, d['test_I'], Vn_te[:, :PRIME],
                 dev).cpu().numpy() * VS - VOFF
    v_true = d['test_V']
    sl = slice(PRIME, None)
    f1 = spike_f1(v_true[:, sl], vp[:, sl])
    fw = f1_by_window(v_true[:, sl], vp[:, sl])
    rest = np.full((1, PRIME), (-65.0 + VOFF) / VS, np.float32)
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    vfi = rollout(model, np.repeat(amps[:, None], T, 1),
                  np.repeat(rest, len(amps), 0),
                  dev).cpu().numpy() * VS - VOFF
    rate = np.array([len(spikes_from_v(x[2000:])) for x in vfi])
    fi = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    vr2 = rollout(model, I2, rest, dev).cpu().numpy()[0] \
        * VS - VOFF
    reb = len(spikes_from_v(vr2[T2 // 2:]))
    return dict(spike_f1=round(f1, 3), f1_by_window=fw,
                fi_rmse_hz=round(fi, 1), rebound_spikes=reb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='ssm',
                    choices=['ssm', 'gru'])
    ap.add_argument('--k', type=int, default=1)
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
    t_idx = np.arange(LAGS[-1], T)
    Wd = np.stack([Vn_tr[:, t_idx - lg] for lg in LAGS], -1)
    Iw = d['train_I'][:, t_idx]
    X = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    Y = torch.tensor(hh_rhs(Str[:, t_idx], Iw)[..., 0],
                     dtype=torch.float32)
    scale = Y.std() + 1e-8
    Vw = torch.tensor(Vn_tr[:, t_idx], dtype=torch.float32)
    Wt = torch.where(Vw > 0.45, 10.0, 1.0)
    torch.manual_seed(args.seed)
    model = JointModel(args.k, args.kind).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            if args.kind == 'ssm':
                c = model.scan(x)
                pred = model.vdot(x, c)
                loss = ((((pred - y) / scale) ** 2) * w_).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                c0 = None
                Tn = x.shape[1]
                for s in range(0, Tn, 500):
                    xs = x[:, s:s + 500]
                    cs = model.gru_pass(
                        xs, None if c0 is None else c0.detach())
                    c0 = cs[:, -1]
                    pred = model.vdot(xs, cs)
                    loss = ((((pred - y[:, s:s + 500]) / scale)
                             ** 2) * w_[:, s:s + 500]).mean()
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
        print(f'joint-{args.kind}{args.k} s={args.seed} '
              f'ep{ep + 1}: val-F1 {vf1:.3f} '
              f'(best {best_vf1:.3f})', flush=True)
    ts = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(model, d, Ste[..., 0], dev)
    nprm = sum(p_.numel() for p_ in model.parameters())
    res = dict(arm=f'joint-{args.kind}{args.k}', seed=args.seed,
               params=nprm, train_seconds=round(ts, 1),
               best_val_f1=round(best_vf1, 3), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'joint_{args.kind}{args.k}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
