"""M13 — INVERTED DAGGER: the training distribution is the only
variable.

Rounds of: generate model-rollout trajectory corpus with the
CURRENT model (gradient-free) -> phase-align teacher labels per
step -> retrain BOTH the scan (A/B) and field on a batch mixture
(p_rollout of batches from rollout sequences, rest teacher).
Rollout sequences train exactly like teacher sequences (parallel
scan + vdot loss), so the state dynamics adapt to the model
distribution.

python3 -m whitebox.hh_invert --p 0.95 --seed 0
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
NW = len(LAGS)
NF = NW + 1
ROLL_H = 1500          # rollout corpus horizon (150 ms)
ROLL_N = 96


STATE_FULL = None   # set in main: full normalized teacher state


def gen_corpus(model, Vn, I_raw, Ydot, dev, rng):
    """Model rollouts -> (X [n, T, NF], Y [n, T]) with
    phase-aligned teacher labels. Gradient-free."""
    B, T = Vn.shape
    starts = rng.integers(PRIME, T - ROLL_H - 1, ROLL_N)
    seqs = rng.integers(0, B, ROLL_N)
    dt_sub = DT * REC_EVERY / SUB
    Xseq = np.zeros((ROLL_N, ROLL_H, NF), np.float32)
    Yseq = np.zeros((ROLL_N, ROLL_H), np.float32)
    with torch.no_grad():
        hist = torch.zeros(ROLL_N, PRIME + ROLL_H)
        for j, (b, t0) in enumerate(zip(seqs, starts)):
            hist[j, :PRIME] = torch.tensor(Vn[b, t0 - PRIME:t0])
        Ib = torch.tensor(np.stack(
            [I_raw[b, t0 - PRIME:t0 + ROLL_H]
             for b, t0 in zip(seqs, starts)]),
            dtype=torch.float32) / IS
        c = hist.new_zeros(ROLL_N, model.k)
        for t in range(PRIME, PRIME + ROLL_H):
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
            Xseq[:, t - PRIME] = x.numpy()
            # phase-aligned label: match on (window, I); label =
            # hh_rhs(teacher state at t*, ROLLOUT current)
            wm = x[:, :NW].numpy()
            i_now = (i_t.squeeze(-1).numpy() * IS)
            for j, (b, t0) in enumerate(zip(seqs, starts)):
                tc = t0 + t - PRIME
                lo = max(tc - 30, LAGS[-1])
                hi = min(tc + 30, T - 1)
                rng_t = np.arange(lo, hi)
                cand = np.stack(
                    [Vn[b, rng_t - lg] for lg in LAGS], -1)
                d_w = np.linalg.norm(cand - wm[j], axis=1)
                d_i = np.abs(I_raw[b, rng_t] - i_now[j]) / IS
                tstar = lo + int(np.argmin(d_w + d_i))
                Yseq[j, t - PRIME] = hh_rhs(
                    STATE_FULL[b, tstar:tstar + 1],
                    np.array([i_now[j]]))[0, 0]
    return (torch.tensor(Xseq), torch.tensor(Yseq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rounds', type=int, default=3)
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
    Xt = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    Yt = torch.tensor(Ydot_full[:, t_idx], dtype=torch.float32)
    scale = Yt.std() + 1e-8
    Wtt = torch.where(torch.tensor(
        Vn_tr[:, t_idx], dtype=torch.float32) > 0.45, 10.0, 1.0)
    Vval = norm_state(d['val_V'], d['val_G'])[..., 0]
    global STATE_FULL
    STATE_FULL = Str
    torch.manual_seed(args.seed)
    model = JointModel(1, 'ssm').to(dev)
    rng = np.random.default_rng(args.seed)
    Xr = Yr = None
    log = []
    best_vf1, best_state = -1.0, None

    def train(epochs, lr):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            nb = B // 32
            for _b in range(nb):
                use_roll = (Xr is not None
                            and rng.random() < args.p)
                if use_roll:
                    j = rng.integers(0, len(Xr), 32)
                    x = Xr[j].to(dev)
                    y = Yr[j].to(dev)
                    w_ = torch.where(x[..., 0] > 0.45, 10.0,
                                     1.0)
                else:
                    j = rng.integers(0, B, 32)
                    x = Xt[j].to(dev)
                    y = Yt[j].to(dev)
                    w_ = Wtt[j].to(dev)
                c = model.scan(x)
                pred = model.vdot(x, c)
                loss = ((((pred - y) / scale) ** 2) * w_).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    t0 = time.time()
    seq_seconds = 0.0
    for r in range(args.rounds + 1):
        if r == 0:
            train(args.epochs0, 1e-3)
        else:
            ts = time.time()
            xs, ys = gen_corpus(model, Vn_tr, d['train_I'],
                                Ydot_full, dev, rng)
            seq_seconds += time.time() - ts
            Xr = xs if Xr is None else torch.cat([Xr, xs])
            Yr = ys if Yr is None else torch.cat([Yr, ys])
            train(args.epochsr, 3e-4)
        vp = rollout(model, d['val_I'], Vval[:, :PRIME],
                     dev).cpu().numpy() * VS - VOFF
        vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                             vp[:, PRIME:]))
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {k_: v_.clone() for k_, v_ in
                          model.state_dict().items()}
        log.append(dict(round=r, val_f1=round(vf1, 3),
                        roll_seqs=0 if Xr is None else len(Xr),
                        seq_s=round(seq_seconds, 1)))
        print('ROUND', json.dumps(log[-1]), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(model, d, Ste[..., 0], dev)
    res = dict(arm=f'invert-v2-p{args.p}', seed=args.seed,
               total_seconds=round(time.time() - t0, 1),
               seq_seconds=round(seq_seconds, 1),
               best_val_f1=round(best_vf1, 3), rounds=log, **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'invertv2_p{args.p}_s{args.seed}'
    torch.save(model.state_dict(), OUT / f'{tag}.pt')
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
