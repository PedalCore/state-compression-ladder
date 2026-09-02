"""M13 E1-E3 — event-decision diagnostics, zero training.

E1: miss autocorrelation P(miss n+1 | miss n) vs P(miss), and
T_recover (time from a missed teacher spike until |V_err| < 10 mV
sustained for 5 ms) — the direct entrainment measurement.
E2: decision-conditioned collision — delay-space near-pairs in
the decision band (-50..-20 mV, matched I): do neighbours agree
on "teacher spikes within H"? Compared against the same statistic
unconditioned.
E3: pre-event field-error anatomy — per teacher spike classified
hit/miss, compare |F_hat - F| (whitened) in the 10 ms pre-event
window between the two classes.

Uses the saved composition stage-0 checkpoint (seed 0).

python3 -m whitebox.hh_event
"""

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_teacher import spikes_from_v                     # noqa
from hh_diag import F as FieldNet                        # noqa
from hh_diag import hh_rhs, norm_state                   # noqa
from hh_comp import roll                                 # noqa

OUT = pathlib.Path('results')
VS, VOFF, IS = 100.0, 65.0, 10.0
TOL = 2.0            # ms spike-match tolerance
H_DEC = 100          # decision horizon, record steps (10 ms)
LAGS = [0, 30, 60, 90, 120]


def match_spikes(v_true, v_pred):
    """Per teacher spike: matched (hit) or not (miss)."""
    st, sp = spikes_from_v(v_true), spikes_from_v(v_pred)
    used = np.zeros(len(sp), bool)
    hits = np.zeros(len(st), bool)
    for k, t in enumerate(st):
        j = np.flatnonzero(~used & (np.abs(sp - t) <= TOL))
        if len(j):
            used[j[0]] = True
            hits[k] = True
    return st, hits


def main():
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Ste = norm_state(d['test_V'], d['test_G'])
    model = FieldNet(256)
    model.load_state_dict(torch.load(
        OUT / 'comp_stage0_s0.pt', weights_only=True))
    tr = roll(model, d['test_I'], Ste[:, 0], 'cpu', 10)
    v_pred = tr[..., 0] * VS - VOFF
    v_true = d['test_V']

    # ---- E1: miss autocorrelation + T_recover ----
    pairs_mm = pairs_m = 0
    miss_tot = spikes_tot = 0
    recov = []
    for b in range(len(v_true)):
        st, hits = match_spikes(v_true[b], v_pred[b])
        spikes_tot += len(st)
        miss_tot += int((~hits).sum())
        for k in range(len(st) - 1):
            if not hits[k]:
                pairs_m += 1
                if not hits[k + 1]:
                    pairs_mm += 1
        err = np.abs(v_pred[b] - v_true[b])
        for k in np.flatnonzero(~hits):
            i0 = int(st[k] / 0.1)
            ok = err[i0:] < 10.0
            run = 0
            t_rec = None
            for i, o in enumerate(ok):
                run = run + 1 if o else 0
                if run >= 50:
                    t_rec = (i - 49) * 0.1
                    break
            if t_rec is not None:
                recov.append(t_rec)
    p_miss = miss_tot / max(spikes_tot, 1)
    p_mm = pairs_mm / max(pairs_m, 1)
    e1 = dict(p_miss=round(p_miss, 3),
              p_miss_given_miss=round(p_mm, 3),
              n_spikes=spikes_tot, n_miss_pairs=pairs_m,
              t_recover_med_ms=round(float(np.median(recov)), 1)
              if recov else None,
              t_recover_p90_ms=round(float(np.quantile(recov,
                                                       0.9)), 1)
              if recov else None)
    print('RESULT-E1', json.dumps(e1), flush=True)

    # ---- E2: decision-conditioned collision (teacher data) ----
    Vtr = d['train_V']
    Itr = d['train_I']
    B, T = Vtr.shape
    spk_next = np.zeros((B, T), bool)
    for b in range(B):
        for t_ms in spikes_from_v(Vtr[b]):
            i0 = int(t_ms / 0.1)
            spk_next[b, max(i0 - H_DEC, 0):i0] = True
    rng = np.random.default_rng(0)
    b_idx = rng.integers(0, B, 400_000)
    t_idx = rng.integers(LAGS[-1], T, 400_000)
    Vt = Vtr[b_idx, t_idx]
    Dm = np.stack([Vtr[b_idx, t_idx - lg] for lg in LAGS],
                  1) / 100.0
    Iv = Itr[b_idx, t_idx]
    Lb = spk_next[b_idx, t_idx]
    out = {}
    for name, mask in (
            ('global', np.ones(len(Vt), bool)),
            ('decision_band', (Vt > -50.0) & (Vt < -20.0))):
        Dm_, Iv_, Lb_ = Dm[mask], Iv[mask], Lb[mask]
        qs = np.quantile(Iv_, np.linspace(0, 1, 9))
        dis, ndis, base = [], 0, []
        for k in range(8):
            m = (Iv_ >= qs[k]) & (Iv_ <= qs[k + 1])
            if m.sum() < 500:
                continue
            Db, Lbb = Dm_[m], Lb_[m]
            r = 0.05 * float(np.mean(Db.std(0)) + 1e-9) \
                * np.sqrt(Db.shape[1])
            tree = cKDTree(Db)
            prs = tree.query_pairs(r, output_type='ndarray')
            if len(prs) > 20000:
                prs = prs[rng.choice(len(prs), 20000,
                                     replace=False)]
            if len(prs) == 0:
                continue
            dis.append((Lbb[prs[:, 0]] != Lbb[prs[:, 1]]).mean())
            ndis += len(prs)
            base.append(2 * Lbb.mean() * (1 - Lbb.mean()))
        out[name] = dict(
            frac_states=round(float(mask.mean()), 4),
            pairs=int(ndis),
            neighbour_disagree=round(float(np.mean(dis)), 4),
            random_disagree=round(float(np.mean(base)), 4),
            ratio=round(float(np.mean(dis) / (np.mean(base)
                                              + 1e-12)), 3))
    print('RESULT-E2', json.dumps(out), flush=True)

    # ---- E3: pre-event field error, hit vs miss ----
    Ste_full = Ste
    scale = None
    errs = {'hit': [], 'miss': []}
    for b in range(len(v_true)):
        st, hits = match_spikes(v_true[b], v_pred[b])
        S = Ste_full[b]
        Ib = d['test_I'][b]
        F_true = hh_rhs(S, Ib)
        with torch.no_grad():
            F_hat = model(
                torch.tensor(S, dtype=torch.float32),
                torch.tensor(Ib[:, None] / IS,
                             dtype=torch.float32)).numpy()
        if scale is None:
            scale = F_true.std(0) + 1e-9
        E = np.linalg.norm((F_hat - F_true) / scale, axis=1)
        for k, t_ms in enumerate(st):
            i0 = int(t_ms / 0.1)
            lo = max(i0 - 100, 0)
            if i0 - lo < 20:
                continue
            errs['hit' if hits[k] else 'miss'].append(
                float(E[lo:i0].mean()))
    e3 = {c: dict(n=len(v), med=round(float(np.median(v)), 4),
                  p90=round(float(np.quantile(v, 0.9)), 4))
          for c, v in errs.items() if v}
    print('RESULT-E3', json.dumps(e3), flush=True)
    json.dump(dict(e1=e1, e2=out, e3=e3),
              open(OUT / 'event_diag.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
