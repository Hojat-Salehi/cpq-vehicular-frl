"""Single loader of what the server observed (attack_log.py). The attack reads only this module:
received updates, the global model, public parameters (clip radius, eps, mechanism) and the labels
in agents.csv (used for the training runs and for scoring only)."""
import csv
import glob
import json
import os

import numpy as np

from frl.privacy import onebit


def load_meta(run_dir):
    with open(os.path.join(run_dir, 'attack_log', 'meta.json')) as f:
        return json.load(f)


def load_agents(run_dir):
    """[(veh_id, origin_label)] in vehicle order."""
    with open(os.path.join(run_dir, 'attack_log', 'agents.csv')) as f:
        return [(r['veh_id'], int(r['origin_label'])) for r in csv.DictReader(f)]


def load_global(run_dir, t):
    """Flattened float32 global model sent in round t."""
    return np.load(os.path.join(run_dir, 'attack_log', 'global', f'round_{t:03d}.npy'))


def _unflatten(flat, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat[i:i + n].reshape(s))
        i += n
    return out


def load_round(run_dir, t, meta=None):
    """{veh: flattened float32 model as the server reconstructs it} for round t.
    cpq / ldpfl: onebit.decode (the server's own function) relative to the global model of round t;
    other methods: the received float weights."""
    meta = meta or load_meta(run_dir)
    shapes = [tuple(s) for s in meta['layer_shapes']]
    up = np.load(os.path.join(run_dir, 'attack_log', 'uploads', f'round_{t:03d}.npz'))
    by_veh = {}
    for key in up.files:
        veh, k = key.rsplit('__', 1)
        by_veh.setdefault(veh, {})[int(k)] = up[key]
    out = {}
    if meta['method'] in onebit.METHODS:
        g = _unflatten(load_global(run_dir, t), shapes)
        radii = onebit.radii_for(g, meta['clip_radius'], meta['layer_names'])
        for veh, arrs in by_veh.items():
            packed = [arrs[k] for k in range(len(arrs))]
            dec = onebit.decode(packed, g, radii, meta['eps'])
            out[veh] = np.concatenate([d.ravel() for d in dec]).astype(np.float32)
    else:
        for veh, arrs in by_veh.items():
            out[veh] = np.concatenate([np.asarray(arrs[k], np.float32).ravel() for k in range(len(arrs))])
    return out


def load_observed_update(run_dir, t, veh, meta=None):
    return load_round(run_dir, t, meta)[veh]


def finished_runs(exp_dir):
    """{(method, eps): [run_dir, ...]} of runs whose attack log is complete."""
    out = {}
    for path in sorted(glob.glob(os.path.join(exp_dir, 'runs', '*', 'attack_log', 'meta.json'))):
        run_dir = os.path.dirname(os.path.dirname(path))
        m = load_meta(run_dir)
        if m['end'] and m['rounds_logged'] == m['rounds']:
            out.setdefault((m['method'], m['eps']), []).append(run_dir)
    return out
