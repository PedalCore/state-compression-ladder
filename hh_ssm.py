"""M13 — mini selective-SSM corrector, FULLY PARALLEL training.

c_{t+1} = a_t c_t + b_t,  a_t = exp(-0.1 / (1 + softplus(w_a x)))
delta_t = eps(V_t) * tanh(w_h c_t + b_h)
Trained by regressing delta onto the frozen field's RESIDUAL
r_t = Vdot_true - f_frozen(window_t, I_t) under teacher windows —
a fixed target sequence; the scan evaluates in closed form via
chunked log-space cumulative sums. No BPTT. ~16 shared params,
one scalar state. Deployment: sequential rollout identical to the
GRU observable hybrid.

python3 -m whitebox.hh_ssm --seed 0
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
from hh_b2 import IS, LAGS, PRIME, VOFF, VS, mlp         # noqa
from hh_obs import evaluate                              # noqa

OUT = pathlib.Path('results')
SUB = 10
SCAN_CHUNK = 250
NF = len(LAGS) + 1              # window + current


class MiniSSM(nn.Module):
    def __init__(self, k=1, eps_hi=0.3):
        super().__init__()
        self.wa = nn.Linear(NF, k)
        self.wb = nn.Linear(NF, k)
        self.head = nn.Linear(k, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.k = k
        self.eps_hi = eps_hi
        self.kind = 'ssm'

    def coeffs(self, X):
        """X (..., NF) -> log_a (..., k) (<0), b (..., k)."""
        tau = 1.0 + torch.nn.functional.softplus(self.wa(X))
        log_a = -0.1 / tau
        b = 0.1 * torch.tanh(self.wb(X))
        return log_a, b

    def scan(self, X, c0=None):
        """Closed-form chunked scan. X (B,T,NF) -> c (B,T,k)."""
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

    def delta_from_c(self, c, Vn):
        """c (..., k), Vn (...) -> delta (...)."""
        eps = 0.05 + (self.eps_hi - 0.05) \
            * (Vn > -0.05).float()
        return eps * torch.tanh(self.head(c).squeeze(-1))

    # interface used by hh_obs rollout/evaluate
    def forward(self, w, i_t, c):
        X = torch.cat([w, i_t], -1)
        log_a, b = self.coeffs(X)
        c = torch.exp(log_a) * c + b
        delta = self.delta_from_c(c, w[:, 0]).unsqueeze(-1)
        return delta, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--eps-hi', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    Vn_tr = Str[..., 0]
    B, T = Vn_tr.shape
    field = mlp(256).to(dev)
    field.load_state_dict(torch.load(
        OUT / f'b2_s{args.seed}.pt', weights_only=True))
    for p_ in field.parameters():
        p_.requires_grad_(False)
    # teacher windows + residual targets, fully vectorized
    t_idx = np.arange(LAGS[-1], T)
    Wd = np.stack([Vn_tr[:, t_idx - lg] for lg in LAGS],
                  -1)                       # (B, T', NW)
    Iw = d['train_I'][:, t_idx]
    X = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    r_true = hh_rhs(Str[:, t_idx], Iw)[..., 0]
    with torch.no_grad():
        f_pred = []
        flat = X.reshape(-1, NF)
        for b0 in range(0, len(flat), 262144):
            f_pred.append(field(flat[b0:b0 + 262144].to(dev))
                          .squeeze(-1).cpu())
        f_pred = torch.cat(f_pred).reshape(X.shape[:2])
    R = torch.tensor(r_true, dtype=torch.float32) - f_pred
    Vw = torch.tensor(Vn_tr[:, t_idx], dtype=torch.float32)
    Wt = torch.where(
        (Vw > -0.05) & (Vw < 0.45), 10.0,
        torch.where(Vw > 0.45, 3.0, 1.0))
    torch.manual_seed(args.seed)
    model = MiniSSM(args.k, args.eps_hi).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    t0 = time.time()
    best_vf1, best_state = -1.0, None
    Vval = norm_state(d['val_V'], d['val_G'])[..., 0]
    for ep in range(args.epochs):
        perm = torch.randperm(B)
        for b0 in range(0, B, 64):
            idx = perm[b0:b0 + 64]
            x = X[idx].to(dev)
            r = R[idx].to(dev)
            w_ = Wt[idx].to(dev)
            c = model.scan(x)
            delta = model.delta_from_c(c, x[..., 0])
            loss = (((delta - r) ** 2) * w_).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        from hh_obs import obs_roll
        vp = obs_roll(field, model, d['val_I'],
                      Vval[:, :PRIME], dev).cpu().numpy() \
            * VS - VOFF
        vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                             vp[:, PRIME:]))
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {k: v.clone() for k, v in
                          model.state_dict().items()}
        print(f'ssm s={args.seed} ep{ep + 1}: val-F1 {vf1:.3f} '
              f'(best {best_vf1:.3f})', flush=True)
    ts = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(field, model, d, Ste[..., 0],
                  np.full((1, PRIME), (-65.0 + VOFF) / VS,
                          np.float32), dev)
    nprm = sum(p_.numel() for p_ in model.parameters())
    res = dict(arm=f'obs-ssm{args.k}-e{args.eps_hi}',
               seed=args.seed, corr_params=nprm,
               train_seconds=round(ts, 1),
               best_val_f1=round(best_vf1, 3), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'ssm{args.k}e{args.eps_hi}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
