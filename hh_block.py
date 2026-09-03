"""M13 — BLOCK-COUPLING (TTT-style) discriminator, CLEAN design.

Both arms: current model generates a b-step rollout, then ONE
update on the IDENTICAL per-step next-V loss with IDENTICAL
targets and optimizer budget. The ONLY difference is stop-
gradient through earlier generated states (detach) vs keeping the
through-time graph (bptt). So "bptt > detach" isolates temporal
credit alone. Fixed rollout-sample budget (--nroll) makes b the
synchronization-interval axis: syncs/epoch = nroll // b.
Invariant: at b=1 there is no earlier generated state, so
detach and bptt must match (built-in sanity check).

python3 -m whitebox.hh_block --mode detach --b 10 --seed 0
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
NSEQ = 32
STATE_FULL = None


def block_update(model, opt, Vn, I_raw, b, dev, rng, mode):
    """One block: current model generates a b-step rollout, then
    ONE update. IDENTICAL per-step next-V loss and targets for
    both arms; the ONLY difference is whether gradient flows
    through earlier generated states:
      bptt   : keep the graph across the block
      detach : stop-gradient each state after it is produced
    Target = teacher next V at the phase-aligned state under the
    ROLLOUT current (current-consistent)."""
    B, T = Vn.shape
    starts = rng.integers(PRIME, T - b - 1, NSEQ)
    seqs = rng.integers(0, B, NSEQ)
    dt_sub = DT * REC_EVERY / SUB
    Ib = torch.tensor(np.stack(
        [I_raw[s, t0 - PRIME:t0 + b]
         for s, t0 in zip(seqs, starts)]),
        dtype=torch.float32, device=dev) / IS
    buf = [torch.tensor(Vn[seqs, starts - PRIME + i],
                        dtype=torch.float32, device=dev)
           for i in range(PRIME)]
    c = buf[0].new_zeros(NSEQ, model.k)
    preds, ys = [], []
    with torch.enable_grad():
        for t in range(b):
            ti = PRIME + t
            lagv = torch.stack(
                [buf[ti - 1 - lg] for lg in LAGS[1:]], 1)
            v = buf[ti - 1]
            i_t = Ib[:, ti - 1:ti]
            x = torch.cat([v[:, None], lagv, i_t], 1)
            c = model.step_c(x, c)
            vnext = v
            for _ in range(SUB):
                vnext = vnext + dt_sub * model.vdot(x, c)
            vnext = torch.clamp(vnext, -0.6, 1.6)
            # SAME target/loss both arms: teacher next-V under
            # rollout current at the phase-aligned state
            tt_arr = starts + t
            y = torch.tensor(
                [Vn[s, min(tt + 1, T - 1)]
                 for s, tt in zip(seqs, tt_arr)],
                dtype=torch.float32, device=dev)
            preds.append(vnext)
            ys.append(y)
            # advance buffer; detach = cut credit through history
            buf.append(vnext.detach() if mode == 'detach'
                       else vnext)
            if mode == 'detach':
                c = c.detach()
    P = torch.stack(preds, 1)
    Y = torch.stack(ys, 1)
    w = torch.where(Y > 0.45, 10.0, 1.0)
    loss = (((P - Y) ** 2) * w).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='detach',
                    choices=['detach', 'bptt'])
    ap.add_argument('--b', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--nroll', type=int, default=400,
                    help='fixed current-policy samples/epoch; '
                         'blocks = nroll // b (sync count)')
    ap.add_argument('--dev', default='cpu')
    args = ap.parse_args()
    global STATE_FULL
    dev = args.dev
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    Ste = norm_state(d['test_V'], d['test_G'])
    STATE_FULL = Str
    Vn_tr = Str[..., 0]
    B, T = Vn_tr.shape
    # parallel teacher batch pool
    t_idx = np.arange(LAGS[-1], T)
    Wd = np.stack([Vn_tr[:, t_idx - lg] for lg in LAGS], -1)
    Iw = d['train_I'][:, t_idx]
    Xt = torch.tensor(np.concatenate(
        [Wd, (Iw / IS)[..., None]], -1), dtype=torch.float32)
    Yt = torch.tensor(hh_rhs(Str[:, t_idx], Iw)[..., 0],
                      dtype=torch.float32)
    scale = Yt.std() + 1e-8
    Vval = norm_state(d['val_V'], d['val_G'])[..., 0]
    torch.manual_seed(args.seed)
    model = JointModel(1, 'ssm').to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    coupled_s = 0.0
    best_vf1, best_state = -1.0, None
    for ep in range(args.epochs):
        # parallel teacher pass
        perm = torch.randperm(B)
        for b0 in range(0, B, 32):
            idx = perm[b0:b0 + 32]
            c = model.scan(Xt[idx].to(dev))
            pred = model.vdot(Xt[idx].to(dev), c)
            loss = (((pred - Yt[idx].to(dev)) / scale)
                    ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        # coupled blocks
        ts = time.time()
        nblk = max(1, args.nroll // args.b)   # fixed sample budget
        for _ in range(nblk):
            block_update(model, opt, Vn_tr, d['train_I'],
                         args.b, dev, rng, args.mode)
        coupled_s += time.time() - ts
        if (ep + 1) % 4 == 0:
            vp = rollout(model, d['val_I'], Vval[:, :PRIME],
                         dev).cpu().numpy() * VS - VOFF
            vf1 = float(spike_f1(d['val_V'][:, PRIME:],
                                 vp[:, PRIME:]))
            if vf1 > best_vf1:
                best_vf1 = vf1
                best_state = {k_: v_.clone() for k_, v_ in
                              model.state_dict().items()}
            print(f'block {args.mode} b={args.b} s={args.seed} '
                  f'ep{ep + 1}: val-F1 {vf1:.3f} '
                  f'(best {best_vf1:.3f})', flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    ev = evaluate(model, d, Ste[..., 0], dev)
    res = dict(arm=f'block-{args.mode}-b{args.b}', seed=args.seed,
               coupled_seconds=round(coupled_s, 1),
               best_val_f1=round(best_vf1, 3), **ev)
    print('RESULT', json.dumps(res), flush=True)
    tag = f'block_{args.mode}_b{args.b}_s{args.seed}'
    json.dump(res, open(OUT / f'{tag}.json', 'w'))


if __name__ == '__main__':
    main()
