"""M13 — history-contamination ladder on the B2 delay field.

Rollout at contamination level g = 0..4: g lag slots (most-recent
first) are filled from the model's own generated history; the
rest come from the teacher. The integrated V_t is always the
model's. F1 vs g separates 'the field is good, self-contamination
kills it' (g=0 high) from 'the map lacks decision precision'
(g=0 low).

python3 -m whitebox.hh_contam [--ckpt b2_s0]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import DT, REC_EVERY                     # noqa
from hh_diag import norm_state                           # noqa
from hh_surrogate import spike_f1                        # noqa
from hh_b2 import IS, LAGS, PRIME, VOFF, VS, mlp         # noqa

OUT = pathlib.Path('results')
SUB = 10


def roll_contam(model, I_mv, v_true_n, g, dev='cpu', bs=32):
    """g = number of lag slots (most-recent first) from model
    history; remaining lags from teacher. Returns V (mV)."""
    outs = []
    dt_sub = DT * REC_EVERY / SUB
    with torch.no_grad():
        for b0 in range(0, len(I_mv), bs):
            Ib = torch.tensor(I_mv[b0:b0 + bs] / IS,
                              dtype=torch.float32, device=dev)
            Vt = torch.tensor(v_true_n[b0:b0 + bs],
                              dtype=torch.float32, device=dev)
            B, T = Ib.shape
            hist = Vt.clone()          # start as teacher; overwrite
            for t in range(PRIME, T):
                lagv = []
                for j, lg in enumerate(LAGS[1:]):
                    src = hist if j < g else Vt
                    lagv.append(src[:, t - 1 - lg])
                lagv = torch.stack(lagv, 1)
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
    ap.add_argument('--ckpt', default='b2_s0')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Ste = norm_state(d['test_V'], d['test_G'])
    Vn = Ste[..., 0]
    model = mlp(256)
    model.load_state_dict(torch.load(OUT / f'{args.ckpt}.pt',
                                     weights_only=True))
    model.eval()
    sl = slice(PRIME, None)
    res = dict(ckpt=args.ckpt)
    for g in range(5):
        vp = roll_contam(model, d['test_I'], Vn, g)
        f1 = spike_f1(d['test_V'][:, sl], vp[:, sl])
        vr = float(np.sqrt(np.mean(
            (vp[:, sl] - d['test_V'][:, sl]) ** 2)))
        res[f'g{g}'] = dict(f1=round(f1, 3), vrmse=round(vr, 1))
        print(f'g={g}: F1 {f1:.3f}  V-RMSE {vr:.1f}', flush=True)
    print('RESULT-CONTAM', json.dumps(res), flush=True)
    json.dump(res, open(OUT / f'contam_{args.ckpt}.json', 'w'))


if __name__ == '__main__':
    main()
