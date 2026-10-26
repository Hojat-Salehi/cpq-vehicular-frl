# mse_log.py
# Empirical aggregation error of the privacy mechanism, per round, for the CPQ-lambda ablation
# (A1). Switched on by FL_MSE_LOG=1 (tools/run_experiment.py). Logging only: it uses arrays
# run_shared.py already has and draws no random numbers, so the run is unchanged.
#
# <out_dir>/mse.csv, one row per round:
#   round, aggregated   number of updates the server averaged
#   mse                 mean over all weights of (server aggregate - reference)^2
#   mse_<layer>         the same per layer
# reference = the base model + the average of the clipped updates clip(w_i - base_i, -r, r) of the
# vehicles the server averaged, i.e. what the server would get without privacy noise (the one-bit
# and additive mechanisms are unbiased for this value). Clipping itself is therefore not counted.

import os

import numpy as np


def enabled():
    return os.environ.get('FL_MSE_LOG') == '1'


class MseLog:
    def __init__(self, out_dir, layer_names, radii_fn):
        self.layer_names = list(layer_names)
        self.radii_fn = radii_fn              # base weights -> per-layer clip radius
        self.path = os.path.join(out_dir, 'mse.csv')
        with open(self.path, 'w') as f:
            f.write(','.join(['round', 'aggregated', 'mse'] + [f'mse_{n}' for n in self.layer_names]) + '\n')
        self.clipped = {}

    def updates(self, local):
        """local: {veh: (trained weights, base weights)} before encoding."""
        self.clipped = {}
        for v, (w, base) in local.items():
            radii = self.radii_fn(base)
            self.clipped[v] = [np.asarray(b, np.float64) + np.clip(np.asarray(x, np.float64) - np.asarray(b, np.float64), -r, r)
                               for x, b, r in zip(w, base, radii)]

    def round(self, rnd, aggregated_vehs, new_global):
        if not aggregated_vehs or new_global is None:
            return
        ref = [np.mean([self.clipped[v][k] for v in aggregated_vehs], axis=0) for k in range(len(new_global))]
        sq = [np.square(np.asarray(g, np.float64) - r) for g, r in zip(new_global, ref)]
        total = sum(float(s.sum()) for s in sq) / sum(s.size for s in sq)
        with open(self.path, 'a') as f:
            f.write(','.join([str(rnd), str(len(aggregated_vehs)), f'{total:.6e}'] +
                             [f'{float(s.mean()):.6e}' for s in sq]) + '\n')
