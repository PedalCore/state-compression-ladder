"""M13 — OBSERVABLE-TRACK HYBRID: the deployment test.

Deterministic delay buffer (shift register) + FROZEN window-field
(B2 checkpoint, F1 0.004 alone) + k-state recurrent corrector,
trust-bounded on observed V, sequentially trained with val-F1
selection. Voltage-only supervision throughout.

  dV/dt = f_frozen(window, I) + eps(V) * tanh(Head(c))
  c <- GRUCell([window, I], c)     (once per 0.1 ms step)

python3 -m whitebox.hh_obs --k 1 --seed 0
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
from hh_diag import norm_state                           # noqa
from hh_surrogate import spike_f1                        # noqa
from hh_b2 import IS, LAGS, PRIME, VOFF, VS, mlp         # noqa
from hh_comp import f1_by_window                         # noqa
from hh_event import match_spikes                        # noqa

OUT = pathlib.Path('results')
SUB = 10
CHUNK = 500
NW = len(LAGS)


class ObsCorrector(nn.Module):
    def __init__(self, k, kind='rec'):
        super().__init__()
        self.kind = kind
        if kind == 'rec':
            self.cell = nn.GRUCell(NW + 1, k)
            self.head = nn.Linear(k, 1)
        else:
            self.net = nn.Sequential(nn.Linear(NW + 1, k),
                                     nn.Tanh())
            self.head = nn.Linear(k, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.k = max(k, 1)

    def forward(self, w, i_t, c):
        if self.kind == 'rec':
            c = self.cell(torch.cat([w, i_t], -1), c)
            raw = self.head(c)
        else:
            raw = self.head(self.net(torch.cat([w, i_t], -1)))
        v = w[:, 0:1]
        eps = 0.05 + 0.25 * (v > -0.05).float()
        return eps * torch.tanh(raw), c


def obs_roll(field, corr, I_mv, v_prime, dev, bs=32,
             grad=False):
    """Standalone rollout: buffer filled with the model's own V
    after priming. Returns normalized V history (B, T)."""
    outs = []
    dt_sub = DT * REC_EVERY / SUB
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            B, T = Ib.shape
            hist = torch.zeros(B, T, device=dev)
            hist[:, :PRIME] = torch.tensor(
                v_prime[b0:b0 + bs], dtype=torch.float32,
                device=dev)
            c = hist.new_zeros(B, corr.k)
            for t in range(PRIME, T):
                lagv = torch.stack(
                    [hist[:, t - 1 - lg] for lg in LAGS[1:]], 1)
                v = hist[:, t - 1]
                i_t = Ib[:, t - 1:t]
                w = torch.cat([v[:, None], lagv], 1)
                delta, c = corr(w, i_t, c)
                for _ in range(SUB):
                    x = torch.cat([v[:, None], lagv, i_t], 1)
                    v = v + dt_sub * (field(x).squeeze(-1)
                                      + delta.squeeze(-1))
                hist[:, t] = torch.clamp(v, -0.6, 1.6)
            outs.append(hist)
    return torch.cat(outs, 0)


def evaluate(field, corr, d, Vn_te, rest_prime, dev):
    vp = obs_roll(field, corr, d['test_I'], Vn_te[:, :PRIME],
                  dev).cpu().numpy() * VS - VOFF
    v_true = d['test_V']
    sl = slice(PRIME, None)
    f1 = spike_f1(v_true[:, sl], vp[:, sl])
    fw = f1_by_window(v_true[:, sl], vp[:, sl])
    trans = dict(cc=0, entry=0, cont=0, recovery=0)
    for b in range(len(v_true)):
        st, hits = match_spikes(v_true[b], vp[b])
        prev = True
        for kk in range(len(st)):
            cls = (('cc' if hits[kk] else 'entry') if prev
                   else ('recovery' if hits[kk] else 'cont'))
            trans[cls] += 1
            prev = hits[kk]
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    vfi = obs_roll(field, corr, np.repeat(amps[:, None], T, 1),
                   np.repeat(rest_prime, len(amps), 0),
                   dev).cpu().numpy() * VS - VOFF
    rate = np.array([len(spikes_from_v(x[2000:])) for x in vfi])
    fi = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    vr2 = obs_roll(field, corr, I2, rest_prime,
                   dev).cpu().numpy()[0] * VS - VOFF
    reb = len(spikes_from_v(vr2[T2 // 2:]))
    return dict(spike_f1=round(f1, 3), f1_by_window=fw,
                entry_rate=round(trans['entry']
                                 / max(trans['cc']
                                       + trans['entry'], 1), 3),
                fi_rmse_hz=round(fi, 1), rebound_spikes=reb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--kind', default='rec',
                    choices=['rec', 'static'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    Vn_tr = torch.tensor(Str[..., 0])
    Vn_te = Ste[..., 0]
    rest_prime = np.full((1, PRIME), (-65.0 + VOFF) / VS,
                         np.float32)
    field = mlp(256).to(dev)
    field.load_state_dict(torch.load(
        OUT / f'b2_s{args.seed}.pt', weights_only=True))
    for p_ in field.parameters():
        p_.requires_grad_(False)
    torch.manual_seed(args.seed)
    corr = ObsCorrector(args.k, args.kind).to(dev)
    opt = torch.optim.Adam(corr.parameters(), lr=1e-3)
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    W = torch.where(Vn_tr > 0.45, 10.0, 1.0)
    B, T = Itr.shape
    dt_sub = DT * REC_EVERY / SUB
    t0 = time.time()
    best_vf1, best_state = -1.0, None
    for ep in range(args.epochs):
        perm = torch.randperm(B)
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            hist = None
            for c0 in range(PRIME, T, CHUNK):
                hi = min(c0 + CHUNK, T)
                if hist is None:
                    hist = Vn_tr[idx, :].clone().to(dev)
                    c = hist.new_zeros(len(idx), corr.k)
                else:
                    c = c.detach()
                x_i = Itr[idx].to(dev)
                y = Vn_tr[idx, c0:hi].to(dev)
                w_ = W[idx, c0:hi].to(dev)
                preds = []
                for t in range(c0, hi):
                    lagv = torch.stack(
                        [hist[:, t - 1 - lg]
                         for lg in LAGS[1:]], 1)
                    v = hist[:, t - 1]
                    i_t = x_i[:, t - 1:t]
                    wv = torch.cat([v[:, None], lagv], 1)
                    delta, c = corr(wv, i_t, c)
                    for _ in range(SUB):
                        xf = torch.cat([v[:, None], lagv, i_t], 1)
                        v = v + dt_sub * (
                            field(xf).squeeze(-1)
                            + delta.squeeze(-1))
                    v = torch.clamp(v, -0.6, 1.6)
                    hist = hist.clone()
                    hist[:, t] = v
                    preds.append(v)
                pred = torch.stack(preds, 1)
                loss = (((pred - y) ** 2) * w_).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    corr.parameters(), 1.0)
                opt.step()
                hist = hist.detach()
        vp = obs_roll(field, corr, d['val_I'],
                      norm_state(d['val_V'],
                                 d['val_G'])[..., 0][:, :PRIME],
                      dev).cpu().numpy() * VS - VOFF
        vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                             vp[:, PRIME:]))
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {kk: v_.clone() for kk, v_ in
                          corr.state_dict().items()}
        print(f'obs k={args.k} s={args.seed} ep{ep + 1}: '
              f'val-F1 {vf1:.3f} (best {best_vf1:.3f})',
              flush=True)
    ts = time.time() - t0
    if best_state is not None:
        corr.load_state_dict(best_state)
    ev = evaluate(field, corr, d, Vn_te, rest_prime, dev)
    nprm = sum(p_.numel() for p_ in corr.parameters())
    res = dict(arm=f'obs-{args.kind}{args.k}',
               seed=args.seed,
               corr_params=nprm,
               train_seconds=round(ts, 1),
               best_val_f1=round(best_vf1, 3), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'obs_{args.kind}{args.k}_s{args.seed}'
    torch.save(corr.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
