"""Single-shot held-out evaluation of the headline finalists on
the LOCKED corpus. Reports the paper's test numbers; writes
results/locked_eval.json."""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hh_hybrid import Corrector, evaluate as hyb_eval            # noqa
from hh_diag import F as FieldNet, norm_state                    # noqa
from hh_teacher import init_state                                # noqa

R = pathlib.Path('results')


def main():
    raw = dict(np.load(R / 'hh_data_final_LOCKED.npz'))
    d = dict(raw)
    d['test_V'], d['test_G'], d['test_I'] = (
        raw['final_V'], raw['final_G'], raw['final_I'])
    Ste = norm_state(d['test_V'], d['test_G'])
    V0, m0, h0, n0 = init_state(1)
    rest = np.array([(V0[0] + 65.) / 100., m0[0], h0[0], n0[0]],
                    np.float32)
    out = {}
    for kc in (8, 1):
        for seed in (0, 1):
            field = FieldNet(256)
            field.load_state_dict(torch.load(
                R / f'comp_stage0_s{seed}.pt', weights_only=True))
            for p_ in field.parameters():
                p_.requires_grad_(False)
            corr = Corrector('rec', kc)
            corr.trust = True
            corr.load_state_dict(torch.load(
                R / f'hyb-rec{kc}-trust-sel-v2_s{seed}.pt',
                weights_only=True))
            ev = hyb_eval(field, corr, d, Ste, rest, 'cpu')
            out[f'hybrid_kc{kc}_s{seed}'] = ev
            print(f'LOCKED hybrid kc={kc} s={seed}: '
                  f'F1 {ev["spike_f1"]}  f-I {ev["fi_rmse_hz"]}  '
                  f'rebound {ev["rebound_spikes"]}', flush=True)
    json.dump(out, open(R / 'locked_eval.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
