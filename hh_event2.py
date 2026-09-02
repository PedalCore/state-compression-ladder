"""M13 E3-v2 — what distinguishes episode-ENTRY misses?

For every teacher spike, features 2 ms before the event: teacher
state (V, m, h, n), input, true dV/dt, and the model's SIGNED
per-component field errors averaged over the window. Events
classed hit / entry (first miss after a hit) / continuation
(miss following a miss). Discriminates the two surviving
hypotheses: badly-ORIENTED field error at entry (signed errors
differ) vs irreducible sensitivity (entries cluster at low-margin
teacher spikes with no error signature).

python3 -m whitebox.hh_event2
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_diag import F as FieldNet                        # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_comp import roll                                 # noqa
from hh_event import match_spikes                        # noqa

OUT = pathlib.Path('results')
VS, VOFF, IS = 100.0, 65.0, 10.0
PRE = 20          # feature window: 2 ms before the event


def main():
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Ste = norm_state(d['test_V'], d['test_G'])
    model = FieldNet(256)
    model.load_state_dict(torch.load(
        OUT / 'comp_stage0_s0.pt', weights_only=True))
    v_pred = roll(model, d['test_I'], Ste[:, 0], 'cpu',
                  10)[..., 0] * VS - VOFF
    v_true = d['test_V']
    rows = []
    for b in range(len(v_true)):
        st, hits = match_spikes(v_true[b], v_pred[b])
        S = Ste[b]
        Ib = d['test_I'][b]
        F_true = hh_rhs(S, Ib)
        with torch.no_grad():
            F_hat = model(
                torch.tensor(S, dtype=torch.float32),
                torch.tensor(Ib[:, None] / IS,
                             dtype=torch.float32)).numpy()
        E = F_hat - F_true                     # signed, (T,4)
        prev_hit = True
        for k, t_ms in enumerate(st):
            i0 = int(t_ms / 0.1)
            lo = i0 - PRE
            if lo < 0:
                prev_hit = hits[k]
                continue
            cls = ('hit' if hits[k] else
                   ('entry' if prev_hit else 'cont'))
            w = slice(lo, i0)
            rows.append(dict(
                cls=cls,
                V=float(S[w, 0].mean() * VS - VOFF),
                m=float(S[w, 1].mean()),
                h=float(S[w, 2].mean()),
                n=float(S[w, 3].mean()),
                I=float(Ib[w].mean()),
                dV_true=float(F_true[w, 0].mean()),
                eV=float(E[w, 0].mean()),
                em=float(E[w, 1].mean()),
                eh=float(E[w, 2].mean()),
                en=float(E[w, 3].mean()),
            ))
            prev_hit = hits[k]
    keys = ['V', 'm', 'h', 'n', 'I', 'dV_true',
            'eV', 'em', 'eh', 'en']
    out = {}
    for cls in ('hit', 'entry', 'cont'):
        sel = [r for r in rows if r['cls'] == cls]
        out[cls] = dict(count=len(sel), **{
            k: round(float(np.median([r[k] for r in sel])), 5)
            for k in keys})
    # simple separation score per feature: |median diff| / IQR
    hits_ = [r for r in rows if r['cls'] == 'hit']
    ent_ = [r for r in rows if r['cls'] == 'entry']
    sep = {}
    for k in keys:
        a = np.array([r[k] for r in hits_])
        b = np.array([r[k] for r in ent_])
        iqr = np.quantile(np.concatenate([a, b]), 0.75) - \
            np.quantile(np.concatenate([a, b]), 0.25) + 1e-12
        sep[k] = round(float(abs(np.median(a) - np.median(b))
                             / iqr), 3)
    out['separation_hit_vs_entry'] = sep
    print('RESULT-E3V2', json.dumps(out), flush=True)
    json.dump(out, open(OUT / 'event2_diag.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
