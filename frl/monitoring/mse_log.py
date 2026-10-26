# Empirical aggregation MSE per round, for the CPQ-lambda ablation (FL_MSE_LOG=1). Logging only.
# <out_dir>/mse.csv: round, number of aggregated updates, and the mean squared difference between the
# server aggregate and the average of the clipped updates (overall and per layer).

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
