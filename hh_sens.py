"""M13 C-track opener — the SENSITIVITY MAP q(x), zero learning,
and sensitivity-weighted field training.

q(x) = lambda_max(sym J_F(x)): the local expansion rate of the
TRUE HH vector field (finite-difference Jacobian of the analytic
RHS at recorded states). Part 1 reports where sensitivity lives
(distribution by voltage band, overlap with the hand-guessed
spike-region weight). Part 2 reruns the A0b field-training config
with w ~ measured lambda+ in place of the hand-guessed 10x
V > -20 mV weight, seeds {0,1}, same metrics — testing whether
the OU-timing F1 plateau (0.736-0.770) was an error-PLACEMENT
problem.

python3 -m whitebox.hh_sens [--diag-only]
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_diag import hh_rhs, norm_state                   # noqa

OUT = pathlib.Path('results')
EPS = 1e-4


def lambda_max_field(S, I):
    """S (...,4) normalized states, I raw current.
    Returns lambda_max of the symmetrized Jacobian, per sample."""
    S = S.reshape(-1, 4)
    I = I.reshape(-1)
    J = np.empty((len(S), 4, 4), np.float32)
    f0 = hh_rhs(S, I)
    for j in range(4):
        Sp = S.copy()
        Sp[:, j] += EPS
        J[:, :, j] = (hh_rhs(Sp, I) - f0) / EPS
    Jsym = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    return np.linalg.eigvalsh(Jsym)[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag-only', action='store_true')
    args = ap.parse_args()
    d = dict(np.load(OUT / 'hh_data_full.npz'))
    Str = norm_state(d['train_V'], d['train_G'])
    rng = np.random.default_rng(0)
    idx = rng.choice(Str.shape[0] * Str.shape[1], 500_000,
                     replace=False)
    S = Str.reshape(-1, 4)[idx]
    I = d['train_I'].reshape(-1)[idx]
    lam = lambda_max_field(S, I)
    V = S[:, 0] * 100.0 - 65.0
    print('lambda_max (per ms) quantiles:',
          np.round(np.quantile(lam, [.5, .9, .99]), 2).tolist(),
          flush=True)
    bands = [(-90, -70), (-70, -60), (-60, -50), (-50, -20),
             (-20, 20), (20, 60)]
    diag = {}
    for lo, hi in bands:
        m = (V >= lo) & (V < hi)
        if m.sum() > 100:
            diag[f'{lo}..{hi}mV'] = dict(
                frac=round(float(m.mean()), 4),
                lam_med=round(float(np.median(lam[m])), 2),
                lam_p95=round(float(np.quantile(lam[m], .95)), 2))
    spikew = V > -20.0
    lamp = np.maximum(lam, 0.0)
    hi_sens = lam > np.quantile(lam, 0.9)
    overlap = float((spikew & hi_sens).sum() / max(hi_sens.sum(),
                                                  1))
    print('RESULT-DIAG', json.dumps(dict(
        bands=diag,
        top10pct_sens_inside_spikeweight=round(overlap, 3))),
        flush=True)
    json.dump(diag, open(OUT / 'sens_diag.json', 'w'), indent=1)
    if args.diag_only:
        return
    # Part 2: sensitivity-weighted A0b-config training.
    import torch
    from hh_diag import main as _unused  # noqa
    Sall = Str.reshape(-1, 4)
    Iall = d['train_I'].reshape(-1)
    lam_all = lambda_max_field(Sall, Iall)
    lamp_all = np.maximum(lam_all, 0.0)
    scale95 = np.quantile(lamp_all, 0.95) + 1e-9
    w_all = 1.0 + 9.0 * np.minimum(lamp_all / scale95, 1.0)
    np.save(OUT / 'sens_weights.npy', w_all.astype(np.float32))
    # reuse hh_diag training loop by monkey-free reimplementation
    from hh_diag import F as FieldNet, rollout
    from hh_teacher import DT, REC_EVERY, init_state, \
        spikes_from_v
    from hh_surrogate import spike_f1
    VS, VOFF, IS = 100.0, 65.0, 10.0
    X = torch.tensor(Sall, dtype=torch.float32)
    Inow = torch.tensor(Iall[:, None] / IS, dtype=torch.float32)
    Y = torch.tensor(hh_rhs(Sall, Iall), dtype=torch.float32)
    scale = Y.std(0, keepdim=True) + 1e-8
    W = torch.tensor(w_all, dtype=torch.float32)
    Ste = norm_state(d['test_V'], d['test_G'])
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = FieldNet(256)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=40, eta_min=1e-4)
        N = len(X)
        for ep in range(40):
            perm = torch.randperm(N)
            for b0 in range(0, N, 4096):
                i2 = perm[b0:b0 + 4096]
                loss = ((((model(X[i2], Inow[i2]) - Y[i2])
                          / scale) ** 2).mean(-1) * W[i2]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            if (ep + 1) % 10 == 0:
                print(f'sensw s={seed} ep{ep + 1}', flush=True)
        v_pred = rollout(model, d['test_I'],
                         np.array([(init_state(1)[0][0] + VOFF)
                                   / VS,
                                   *[g[0] for g in
                                     init_state(1)[1:]]],
                                  np.float32), 'deriv',
                         'cpu')[..., 0] * VS - VOFF
        v_true = Ste[..., 0] * VS - VOFF
        f1 = spike_f1(v_true, v_pred)
        vr = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
        amps = d['fi_amps']
        T = int(1200.0 / (DT * REC_EVERY))
        rest = np.array([(init_state(1)[0][0] + VOFF) / VS,
                         *[g[0] for g in init_state(1)[1:]]],
                        np.float32)
        vfi = rollout(model, np.repeat(amps[:, None], T, 1), rest,
                      'deriv', 'cpu')[..., 0] * VS - VOFF
        rate = np.array([len(spikes_from_v(x[2000:]))
                         for x in vfi])
        fi = float(np.sqrt(np.mean((rate - d['fi_rate']) ** 2)))
        T2 = int(400.0 / (DT * REC_EVERY))
        I2 = np.zeros((1, T2))
        I2[0, :T2 // 2] = -3.0
        vr2 = rollout(model, I2, rest, 'deriv',
                      'cpu')[0, :, 0] * VS - VOFF
        reb = len(spikes_from_v(vr2[T2 // 2:]))
        res = dict(arm='sensw-field', seed=seed,
                   v_rmse_mv=round(vr, 2), spike_f1=round(f1, 3),
                   fi_rmse_hz=round(fi, 1), rebound_spikes=reb)
        print('RESULT', json.dumps(res), flush=True)
        json.dump(res, open(OUT / f'sensw_s{seed}.json', 'w'))


if __name__ == '__main__':
    main()
