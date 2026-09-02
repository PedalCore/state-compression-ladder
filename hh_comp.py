"""M13 — the COMPOSITION experiment: short-horizon multiple-
shooting curriculum on the x-space field.

Stage 0: iid field training (A0b config), checkpoint saved.
Stage 0.5: integrator isolation — same weights under Euler-10 vs
RK4 at 0.1 ms.
Stages 1-4: segment curriculum H = 5/10/20/40 ms from TEACHER
start states (parallel across segments; BPTT only inside), full-
state trajectory loss. After each stage: F1, TTD, F1-by-time-
window (drift discriminator), f-I, rebound.

python3 -m whitebox.hh_comp --seed 0
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import (DT, REC_EVERY, init_state,       # noqa
                                 spikes_from_v)
from hh_diag import F as FieldNet                        # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_surrogate import spike_f1                        # noqa
from hh_b2op import time_to_divergence                   # noqa

OUT = pathlib.Path('results')
VS, VOFF, IS = 100.0, 65.0, 10.0
STAGES = [50, 100, 200, 400]        # record steps (5-40 ms)
TRAIN_SUB = 5
EVAL_SUB = 10


def roll(model, I_mv, s0, dev, sub, method='euler', bs=32):
    """s0 (B,4) normalized start states -> states (B,T,4)."""
    model.eval()
    outs = []
    dt = DT * REC_EVERY / sub
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            s = torch.tensor(s0[b0:b0 + bs], dtype=torch.float32,
                             device=dev)
            traj = []
            for t in range(Ib.shape[1]):
                i_t = Ib[:, t:t + 1]
                for _ in range(sub):
                    if method == 'euler':
                        s = s + dt * model(s, i_t)
                    else:                       # rk4
                        k1 = model(s, i_t)
                        k2 = model(s + 0.5 * dt * k1, i_t)
                        k3 = model(s + 0.5 * dt * k2, i_t)
                        k4 = model(s + dt * k3, i_t)
                        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3
                                          + k4)
                s = torch.clamp(s, -0.5, 1.5)
                traj.append(s)
            outs.append(torch.stack(traj, 1).cpu().numpy())
    return np.concatenate(outs, 0)


def f1_by_window(v_true, v_pred, nw=4):
    T = v_true.shape[1]
    return [round(spike_f1(v_true[:, i * T // nw:(i + 1) * T
                                  // nw],
                           v_pred[:, i * T // nw:(i + 1) * T
                                  // nw]), 3)
            for i in range(nw)]


def full_eval(model, d, Ste, rest, dev, method='euler',
              sub=EVAL_SUB):
    tr = roll(model, d['test_I'], Ste[:, 0], dev, sub, method)
    v_pred = tr[..., 0] * VS - VOFF
    v_true = Ste[..., 0] * VS - VOFF
    f1 = spike_f1(v_true, v_pred)
    vr = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
    fw = f1_by_window(v_true, v_pred)
    ttd = time_to_divergence(v_pred, v_true)
    amps = d['fi_amps']
    T = int(1200.0 / (DT * REC_EVERY))
    vfi = roll(model, np.repeat(amps[:, None], T, 1),
               np.repeat(rest[None], len(amps), 0), dev, sub,
               method)[..., 0] * VS - VOFF
    rate = np.array([len(spikes_from_v(x[2000:])) for x in vfi])
    fi = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
    T2 = int(400.0 / (DT * REC_EVERY))
    I2 = np.zeros((1, T2))
    I2[0, :T2 // 2] = -3.0
    vr2 = roll(model, I2, rest[None], dev, sub,
               method)[0, :, 0] * VS - VOFF
    reb = len(spikes_from_v(vr2[T2 // 2:]))
    return dict(spike_f1=round(f1, 3), v_rmse_mv=round(vr, 2),
                f1_by_window=fw, ttd_ms=round(ttd, 1),
                fi_rmse_hz=round(fi, 1), rebound_spikes=reb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs0', type=int, default=40)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    B, T = Str.shape[:2]
    # ---- stage 0: iid field (A0b config) ----
    Sall = Str.reshape(-1, 4)
    Iall = d['train_I'].reshape(-1)
    X = torch.tensor(Sall, dtype=torch.float32)
    Inow = torch.tensor(Iall[:, None] / IS, dtype=torch.float32)
    Y = torch.tensor(hh_rhs(Sall, Iall), dtype=torch.float32)
    scale = Y.std(0, keepdim=True) + 1e-8
    W = torch.where(torch.tensor(Sall[:, 0]) > 0.45, 10.0, 1.0)
    torch.manual_seed(args.seed)
    model = FieldNet(256).to(dev)
    N = len(X)
    t0 = time.time()
    ck = OUT / f'comp_stage0_s{args.seed}.pt'
    if ck.exists():
        model.load_state_dict(torch.load(ck, weights_only=True))
        print(f'stage0: reusing {ck.name}', flush=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs0, eta_min=1e-4)
        for ep in range(args.epochs0):
            perm = torch.randperm(N)
            for b0 in range(0, N, 4096):
                i2 = perm[b0:b0 + 4096]
                loss = ((((model(X[i2].to(dev), Inow[i2].to(dev))
                           - Y[i2].to(dev)) / scale.to(dev)) ** 2)
                        .mean(-1) * W[i2].to(dev)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        torch.save(model.state_dict(), ck)
    e0 = full_eval(model, d, Ste, rest, dev)
    print('RESULT-STAGE',
          json.dumps(dict(stage=0, seed=args.seed,
                          secs=round(time.time() - t0, 1), **e0)),
          flush=True)
    # ---- stage 0.5: integrator isolation ----
    erk = full_eval(model, d, Ste, rest, dev, method='rk4', sub=1)
    print('RESULT-INTEG',
          json.dumps(dict(seed=args.seed, method='rk4-0.1ms',
                          **erk)), flush=True)
    # ---- stages 1-4: segment curriculum ----
    rng = np.random.default_rng(args.seed)
    dt_tr = DT * REC_EVERY / TRAIN_SUB
    for si, H in enumerate(STAGES, 1):
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        ts = time.time()
        for ep in range(3):
            for _ in range(32):            # 32 x 128 segments
                b_i = rng.integers(0, B, 128)
                t_i = rng.integers(0, T - H - 1, 128)
                s = torch.tensor(Str[b_i, t_i],
                                 dtype=torch.float32, device=dev)
                seg_y = torch.tensor(
                    np.stack([Str[b, t + 1:t + 1 + H]
                              for b, t in zip(b_i, t_i)]),
                    dtype=torch.float32, device=dev)
                seg_i = torch.tensor(
                    np.stack([d['train_I'][b, t:t + H]
                              for b, t in zip(b_i, t_i)]),
                    dtype=torch.float32, device=dev) / IS
                w = torch.where(seg_y[..., 0] > 0.45, 10.0, 1.0)
                preds = []
                for t in range(H):
                    i_t = seg_i[:, t:t + 1]
                    for _ in range(TRAIN_SUB):
                        s = s + dt_tr * model(s, i_t)
                    preds.append(s)
                pred = torch.stack(preds, 1)
                l_seg = (((pred - seg_y) ** 2).mean(-1)
                         * w).mean()
                ia = torch.randint(0, len(X), (4096,))
                l_anchor = ((((model(X[ia].to(dev),
                                     Inow[ia].to(dev))
                               - Y[ia].to(dev)) / scale.to(dev))
                             ** 2).mean(-1) * W[ia].to(dev)).mean()
                loss = l_seg + l_anchor
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                opt.step()
        ev = full_eval(model, d, Ste, rest, dev)
        print('RESULT-STAGE',
              json.dumps(dict(stage=si, horizon_ms=H * 0.1,
                              seed=args.seed,
                              secs=round(time.time() - ts, 1),
                              **ev)), flush=True)
        torch.save(model.state_dict(),
                   OUT / f'comp_stage{si}_s{args.seed}.pt')


if __name__ == '__main__':
    main()
