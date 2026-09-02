"""M13 — HYBRID: frozen parallel-trained field + tiny
sequentially-trained correction state.

dx/dt = F_frozen(x, I) + Head(c),   c <- GRUCell([x, I], c)
(c updated once per 0.1 ms record step; Head zero-initialized so
training step 0 IS the stage-0 baseline). Only the corrector
trains (TBPTT, full-state loss, spike-weighted). The corrector's
measured job: counteract inter-spike under-excitability drift and
soften the inherited decision geometry.

python3 -m whitebox.hh_hybrid --seed 0
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

from hh_teacher import (DT, REC_EVERY, init_state,       # noqa
                                 spikes_from_v)
from hh_diag import F as FieldNet                        # noqa
from hh_diag import norm_state                           # noqa
from hh_comp import f1_by_window                         # noqa
from hh_event import match_spikes                        # noqa
from hh_b2op import time_to_divergence                   # noqa

OUT = pathlib.Path('results')
VS, VOFF, IS = 100.0, 65.0, 10.0
KC = 8
SUB = 10
CHUNK = 500


class Corrector(nn.Module):
    """kind='rec': GRUCell with kc states. kind='static': memoryless
    MLP of matched parameter budget (the capacity control)."""

    def __init__(self, kind='rec', kc=8):
        super().__init__()
        self.kind = kind
        self.kc = kc
        if kind == 'rec':
            self.cell = nn.GRUCell(5, kc)
            self.head = nn.Linear(kc, 4)
        else:
            self.net = nn.Sequential(nn.Linear(5, 24), nn.Tanh())
            self.head = nn.Linear(24, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, i_t, c):
        if self.kind == 'rec':
            c = self.cell(torch.cat([x, i_t], -1), c)
            return self.head(c), c
        z = self.net(torch.cat([x, i_t], -1))
        return self.head(z), c


def hybrid_roll(field, corr, I_mv, s0, dev, bs=32,
                grad=False, want_delta=False):
    outs, douts = [], []
    dt = DT * REC_EVERY / SUB
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            s = torch.tensor(s0[b0:b0 + bs],
                             dtype=torch.float32, device=dev)
            c = s.new_zeros(len(s), max(corr.kc, 1))
            traj, dtraj = [], []
            for t in range(Ib.shape[1]):
                i_t = Ib[:, t:t + 1]
                delta, c = corr(s, i_t, c)
                for _ in range(SUB):
                    s = s + dt * (field(s, i_t) + delta)
                s = torch.clamp(s, -0.5, 1.5)
                traj.append(s)
                if want_delta:
                    dtraj.append(delta.norm(dim=-1))
            outs.append(torch.stack(traj, 1))
            if want_delta:
                douts.append(torch.stack(dtraj, 1))
    if want_delta:
        return torch.cat(outs, 0), torch.cat(douts, 0)
    return torch.cat(outs, 0)


def evaluate(field, corr, d, Ste, rest, dev):
    tr, dn = hybrid_roll(field, corr, d['test_I'], Ste[:, 0],
                         dev, want_delta=True)
    tr = tr.cpu().numpy()
    dn = dn.cpu().numpy()
    v_pred = tr[..., 0] * VS - VOFF
    v_true = d['test_V']
    f1 = float(__import__('whitebox.hh_surrogate',
                          fromlist=['spike_f1'])
               .spike_f1(v_true, v_pred))
    fw = f1_by_window(v_true, v_pred)
    trans = dict(cc=0, entry=0, cont=0, recovery=0)
    for b in range(len(v_true)):
        st, hits = match_spikes(v_true[b], v_pred[b])
        prev = True
        for k in range(len(st)):
            cls = (('cc' if hits[k] else 'entry') if prev
                   else ('recovery' if hits[k] else 'cont'))
            trans[cls] += 1
            prev = hits[k]
    entry_rate = trans['entry'] / max(trans['cc']
                                      + trans['entry'], 1)
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    vfi = hybrid_roll(field, corr,
                      np.repeat(amps[:, None], T, 1),
                      np.repeat(rest[None], len(amps), 0),
                      dev).cpu().numpy()[..., 0] * VS - VOFF
    rate = np.array([len(spikes_from_v(x[2000:])) for x in vfi])
    fi = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    vr2 = hybrid_roll(field, corr, I2, rest[None],
                      dev).cpu().numpy()[0, :, 0] * VS - VOFF
    reb = len(spikes_from_v(vr2[T2 // 2:]))
    Vmv = tr[..., 0] * VS - VOFF
    dbands = {}
    for lo, hi in ((-90, -70), (-70, -50), (-50, -20), (-20, 60)):
        m = (Vmv >= lo) & (Vmv < hi)
        if m.sum() > 100:
            dbands[f'{lo}..{hi}'] = round(
                float(np.median(dn[m])), 4)
    return dict(spike_f1=round(f1, 3), f1_by_window=fw,
                entry_rate=round(entry_rate, 3),
                cont_frac=round(trans['cont']
                                / max(trans['entry']
                                      + trans['cont'], 1), 3),
                fi_rmse_hz=round(fi, 1), rebound_spikes=reb,
                delta_by_band=dbands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kind', default='rec',
                    choices=['rec', 'static'])
    ap.add_argument('--kc', type=int, default=8)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    field = FieldNet(256).to(dev)
    field.load_state_dict(torch.load(
        OUT / f'comp_stage0_s{args.seed}.pt',
        weights_only=True))
    for p_ in field.parameters():
        p_.requires_grad_(False)
    torch.manual_seed(args.seed)
    corr = Corrector(args.kind, args.kc).to(dev)
    opt = torch.optim.Adam(corr.parameters(), lr=1e-3)
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    Ytr = torch.tensor(Str)
    W = torch.where(Ytr[..., 0] > 0.45, 10.0, 1.0)
    B, T = Itr.shape
    dt = DT * REC_EVERY / SUB
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(B)
        tot = cnt = 0.0
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            s = c = None
            for c0 in range(0, T, CHUNK):
                x = Itr[idx, c0:c0 + CHUNK].to(dev)
                y = Ytr[idx, c0:c0 + CHUNK].to(dev)
                w = W[idx, c0:c0 + CHUNK].to(dev)
                if s is None:
                    s = y[:, 0]
                    c = s.new_zeros(len(s), max(corr.kc, 1))
                else:
                    s = s.detach()
                    c = c.detach()
                preds = []
                for t in range(x.shape[1]):
                    i_t = x[:, t:t + 1]
                    delta, c = corr(s, i_t, c)
                    for _ in range(SUB):
                        s = s + dt * (field(s, i_t) + delta)
                    s = torch.clamp(s, -0.5, 1.5)
                    preds.append(s)
                pred = torch.stack(preds, 1)
                loss = (((pred - y) ** 2).mean(-1) * w).mean() \
                    + args.lam * (delta ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    corr.parameters(), 1.0)
                opt.step()
                s = s.detach()
                c = c.detach()
                tot += float(loss) * x.numel()
                cnt += x.numel()
        if (ep + 1) % 3 == 0:
            ev = evaluate(field, corr, d, Ste, rest, dev)
            print(f'hyb s={args.seed} ep{ep + 1}: '
                  f'{json.dumps(ev)}', flush=True)
    ts = time.time() - t0
    ev = evaluate(field, corr, d, Ste, rest, dev)
    nprm = sum(p_.numel() for p_ in corr.parameters())
    arm = (f'hyb-{args.kind}'
           + (f'{args.kc}' if args.kind == 'rec' else ''))
    res = dict(arm=arm, seed=args.seed, corr_params=nprm,
               train_seconds=round(ts, 1), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'{arm}_s{args.seed}'
    torch.save(corr.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
