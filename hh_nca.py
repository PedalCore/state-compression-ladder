"""M13 NCA arm — a 1-D neural cellular automaton over the
certified delay geometry, with optional persistence-pool training.

5 cells x 8 channels; cell i's V-channel initialized from the
teacher voltage at lag 3i ms at prime time; thereafter the state
evolves purely under ONE shared local rule
  s_i <- s_i + f(s_{i-1}, s_i, s_{i+1}, I)
per 0.1 ms step (zero-padded ends). Prediction = V-channel of
cell 0. History maintenance is the rule's job. Arm C adds a
persistence pool: 30% of training chunks start from stored
model-generated states (20% of those with one cell's hidden
channels zeroed) — teach recovery, not just correctness.

python3 -m whitebox.hh_nca --arm b|c --seed 0
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
from hh_b2op import time_to_divergence                   # noqa

OUT = pathlib.Path('results')
IS = 10.0
VS, VOFF = 100.0, 65.0
NC, CH = 5, 8                  # cells, channels
LAG = 30                       # record steps between cells (3 ms)
PRIME = 121
CHUNK = 1000


class NCA(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3 * CH + 1, width),
                               nn.Tanh(), nn.Linear(width, CH))

    def step(self, S, i_t):
        """S (B, NC, CH), i_t (B, 1) -> next S."""
        z = torch.zeros_like(S[:, :1])
        left = torch.cat([z, S[:, :-1]], 1)
        right = torch.cat([S[:, 1:], z], 1)
        inp = torch.cat(
            [left, S, right,
             i_t[:, None, :].expand(-1, NC, 1)], -1)
        return S + self.f(inp)

    def init_state(self, v_hist):
        """v_hist (B, PRIME) normalized V -> S (B, NC, CH)."""
        S = v_hist.new_zeros(len(v_hist), NC, CH)
        for i in range(NC):
            S[:, i, 0] = v_hist[:, -1 - i * LAG]
        return S


def rollout(model, I_mv, v_prime, dev, bs=32):
    model.eval()
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            B, T = Ib.shape
            vp = torch.tensor(v_prime[b0:b0 + bs],
                              dtype=torch.float32, device=dev)
            S = model.init_state(vp)
            vs = [vp[:, t] for t in range(PRIME)]
            for t in range(PRIME, T):
                S = model.step(S, Ib[:, t - 1:t])
                S = torch.clamp(S, -5.0, 5.0)
                vs.append(S[:, 0, 0])
            outs.append(torch.stack(vs, 1).cpu().numpy()
                        * VS - VOFF)
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['b', 'c'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Vn = torch.tensor(Str[..., 0])
    Itr = torch.tensor(d['train_I'] / IS, dtype=torch.float32)
    W = torch.where(Vn > 0.45, 10.0, 1.0)
    B, T = Vn.shape
    torch.manual_seed(args.seed)
    model = NCA().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=2e-4)
    pool = []                  # (state, seq idx, time idx)
    rngp = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(B)
        tot = cnt = 0.0
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            S = None
            for c0 in range(PRIME, T, CHUNK):
                x = Itr[idx, c0 - 1:min(c0 + CHUNK, T) - 1].to(dev)
                y = Vn[idx, c0:c0 + CHUNK].to(dev)
                w = W[idx, c0:c0 + CHUNK].to(dev)
                if S is None:
                    S = model.init_state(
                        Vn[idx, c0 - PRIME:c0].to(dev))
                else:
                    S = S.detach()
                if args.arm == 'c' and rngp.random() < 0.3:
                    cand = [e for e in pool if e[2] == c0]
                    if cand:
                        Sp, pi, _ = cand[
                            rngp.integers(0, len(cand))]
                        S = Sp.to(dev).clone()
                        if rngp.random() < 0.2:
                            ci = rngp.integers(0, NC)
                            S[:, ci, 1:] = 0.0       # damage
                        idx = pi
                        y = Vn[idx, c0:c0 + CHUNK].to(dev)
                        w = W[idx, c0:c0 + CHUNK].to(dev)
                        x = Itr[idx, c0 - 1:
                                min(c0 + CHUNK, T) - 1].to(dev)
                vs = []
                for t in range(x.shape[1]):
                    S = model.step(S, x[:, t:t + 1])
                    S = torch.clamp(S, -5.0, 5.0)
                    vs.append(S[:, 0, 0])
                pred = torch.stack(vs, 1)
                loss = (((pred - y) ** 2) * w).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                S = S.detach()
                if args.arm == 'c' and rngp.random() < 0.1:
                    nxt = min(c0 + CHUNK, T)
                    if nxt < T and len(pool) < 200:
                        pool.append((S.cpu().clone(), idx, nxt))
                tot += float(loss) * pred.numel()
                cnt += pred.numel()
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f'nca-{args.arm} s={args.seed} ep{ep + 1}: '
                  f'loss {tot / cnt:.5f}', flush=True)
    ts = time.time() - t0
    Ste = norm_state(d['test_V'], d['test_G'])
    v_pred = rollout(model, d['test_I'],
                     Ste[..., 0][:, :PRIME], dev)
    sl = slice(PRIME, None)
    f1 = spike_f1(d['test_V'][:, sl], v_pred[:, sl])
    vr = float(np.sqrt(np.mean(
        (v_pred[:, sl] - d['test_V'][:, sl]) ** 2)))
    ttd = time_to_divergence(v_pred, d['test_V'])
    rest = np.full((1, PRIME), (-65.0 + VOFF) / VS, np.float32)
    amps = d['fi_amps']
    Tfi = int(1200.0 / (DT * REC_EVERY))
    vfi = rollout(model, np.repeat(amps[:, None], Tfi, 1),
                  np.repeat(rest, len(amps), 0), dev)
    rate = np.array([len(spikes_from_v(x[2000:])) for x in vfi])
    fi = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    vr2 = rollout(model, I2, rest, dev)[0]
    reb = len(spikes_from_v(vr2[T2 // 2:]))
    nprm = sum(p.numel() for p in model.parameters())
    res = dict(arm=f'nca-{args.arm}', seed=args.seed, params=nprm,
               train_seconds=round(ts, 1),
               v_rmse_mv=round(vr, 2), spike_f1=round(f1, 3),
               ttd_ms=round(ttd, 1), fi_rmse_hz=round(fi, 1),
               rebound_spikes=reb)
    print('RESULT', json.dumps(res), flush=True)
    torch.save(model.state_dict(),
               OUT / f'nca_{args.arm}_s{args.seed}.pt')
    json.dump(res, open(
        OUT / f'nca_{args.arm}_s{args.seed}.json', 'w'))


if __name__ == '__main__':
    main()
