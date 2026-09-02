"""M13 — trajectory-divergence atlas. Instrument, not fix.

For each teacher spike opportunity, classified by transition
(cc / entry / cont / recovery), walk backward tau = 2..50 ms and
record per-component CLOCK-ALIGNED rollout-vs-teacher divergence,
PHASE-ALIGNED distance (nearest teacher state within +-3 ms), and
divergence amplification A(tau) = |dx(t0)|/|dx(t0-tau)|.
Discriminability D_j(tau) = (mu_entry - mu_cc)/pooled std.
Also the transition-probability table per checkpoint.

python3 -m whitebox.hh_atlas [--ckpts comp_stage0_s0,dir_s0,dir_s1]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_diag import F as FieldNet                        # noqa
from hh_diag import norm_state                           # noqa
from hh_comp import roll                                 # noqa
from hh_event import match_spikes                        # noqa

OUT = pathlib.Path('results')
VS, VOFF = 100.0, 65.0
TAUS = [20, 50, 100, 200, 300, 500]     # record steps (2-50 ms)
PH_W = 30                               # phase window +-3 ms
COMP = ['V', 'm', 'h', 'n']


def analyze(ckpt, d, Ste):
    model = FieldNet(256)
    model.load_state_dict(torch.load(OUT / f'{ckpt}.pt',
                                     weights_only=True))
    tr = roll(model, d['test_I'], Ste[:, 0], 'cpu', 10)
    v_pred = tr[..., 0] * VS - VOFF
    v_true = d['test_V']
    rows = []
    trans = dict(cc=0, entry=0, cont=0, recovery=0)
    for b in range(len(v_true)):
        st, hits = match_spikes(v_true[b], v_pred[b])
        prev = True
        for k, t_ms in enumerate(st):
            i0 = int(t_ms / 0.1)
            cls = (('cc' if hits[k] else 'entry') if prev
                   else ('recovery' if hits[k] else 'cont'))
            trans[cls] += 1
            feats = {}
            for tau in TAUS:
                i = i0 - tau
                if i < 0:
                    continue
                dx = tr[b, i] - Ste[b, i]
                for j, c in enumerate(COMP):
                    feats[f'd{c}_{tau}'] = float(dx[j])
                feats[f'norm_{tau}'] = float(
                    np.linalg.norm(dx))
                lo, hi = max(i - PH_W, 0), i + PH_W
                ph = float(np.min(np.linalg.norm(
                    Ste[b, lo:hi] - tr[b, i], axis=1)))
                feats[f'phase_{tau}'] = ph
            n0 = np.linalg.norm(tr[b, i0] - Ste[b, i0])
            for tau in (100, 300):
                i = i0 - tau
                if i >= 0:
                    nl = np.linalg.norm(tr[b, i] - Ste[b, i])
                    feats[f'amp_{tau}'] = float(
                        n0 / (nl + 1e-9))
            rows.append(dict(cls=cls, **feats))
            prev = hits[k]
    n_ev = sum(trans.values())
    tot_miss = trans['entry'] + trans['cont']
    table = dict(**trans,
                 entry_rate=round(trans['entry']
                                  / max(trans['cc']
                                        + trans['entry'], 1), 3),
                 cont_frac=round(trans['cont']
                                 / max(tot_miss, 1), 3),
                 f1_events=round((n_ev - tot_miss) / max(n_ev, 1),
                                 3))
    # discriminability entry vs cc per feature
    cc = [r for r in rows if r['cls'] == 'cc']
    en = [r for r in rows if r['cls'] == 'entry']
    disc = {}
    for key in sorted({k for r in rows for k in r if k != 'cls'}):
        a = np.array([r[key] for r in cc if key in r])
        b_ = np.array([r[key] for r in en if key in r])
        if len(a) < 20 or len(b_) < 20:
            continue
        pool = np.std(np.concatenate([a, b_])) + 1e-12
        disc[key] = round(float((np.mean(b_) - np.mean(a))
                                / pool), 3)
    top = sorted(disc.items(), key=lambda kv: -abs(kv[1]))[:12]
    res = dict(ckpt=ckpt, transitions=table,
               top_discriminators=dict(top))
    print('RESULT-ATLAS', json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts',
                    default='comp_stage0_s0,dir_s0,dir_s1')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Ste = norm_state(d['test_V'], d['test_G'])
    results = [analyze(c, d, Ste)
               for c in args.ckpts.split(',')]
    json.dump(results, open(OUT / 'atlas_results.json', 'w'),
              indent=1)


if __name__ == '__main__':
    main()
