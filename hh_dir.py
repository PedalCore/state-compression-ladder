"""M13 — the DIRECTIONAL FIX: gate-component weighting in the
measured decision band.

From-scratch A0b protocol (deriv field, width 256, 40 epochs,
standard 10x spike-region weight), plus per-COMPONENT weights in
the decision band (V in -50..-20 mV): m-error x10, h/n x5 —
aimed at the E3-v2 signature (dm/dt underestimation at low-margin
decisions). Post-hoc mechanistic closure: the episode-entry
anatomy rerun on the trained model.

python3 -m whitebox.hh_dir --seed 0
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
from hh_comp import f1_by_window, full_eval              # noqa

OUT = pathlib.Path('results')
VS, VOFF, IS = 100.0, 65.0, 10.0
BAND = (0.15, 0.45)          # normalized V: -50..-20 mV
W_M, W_HN = 10.0, 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + VOFF) / VS, m0[0], h0[0], n0[0]],
                    np.float32)
    Sall = Str.reshape(-1, 4)
    Iall = d['train_I'].reshape(-1)
    X = torch.tensor(Sall, dtype=torch.float32)
    Inow = torch.tensor(Iall[:, None] / IS, dtype=torch.float32)
    Y = torch.tensor(hh_rhs(Sall, Iall), dtype=torch.float32)
    scale = Y.std(0, keepdim=True) + 1e-8
    Wrow = torch.where(X[:, 0] > 0.45, 10.0, 1.0)   # scalar part
    in_band = (X[:, 0] > BAND[0]) & (X[:, 0] < BAND[1])
    Wcomp = torch.ones(len(X), 4)
    Wcomp[in_band, 1] = W_M
    Wcomp[in_band, 2] = W_HN
    Wcomp[in_band, 3] = W_HN
    print(f'decision band: {float(in_band.float().mean()):.4f} '
          f'of states', flush=True)
    torch.manual_seed(args.seed)
    model = FieldNet(256).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4)
    N = len(X)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        for b0 in range(0, N, 4096):
            i2 = perm[b0:b0 + 4096]
            err = (model(X[i2].to(dev), Inow[i2].to(dev))
                   - Y[i2].to(dev)) / scale.to(dev)
            loss = ((err ** 2 * Wcomp[i2].to(dev)).mean(-1)
                    * Wrow[i2].to(dev)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f'dir s={args.seed} ep{ep + 1}', flush=True)
    ts = time.time() - t0
    ev = full_eval(model, d, Ste, rest, dev)
    res = dict(arm='dir-field', seed=args.seed,
               train_seconds=round(ts, 1), **ev)
    print('RESULT', json.dumps(res), flush=True)
    torch.save(model.state_dict(),
               OUT / f'dir_s{args.seed}.pt')
    json.dump(res, open(OUT / f'dir_s{args.seed}.json', 'w'))


if __name__ == '__main__':
    main()
