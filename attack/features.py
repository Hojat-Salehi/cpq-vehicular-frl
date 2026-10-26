"""Attack features per (round, vehicle), from u = received update - global model:
  pos    fc1 weights of the position inputs (current edge and destination x, y)
  full   the whole update (reduced with PCA later)
  probe  Q-value change on the 60 probe states of attack/probe_states.npy
Cached in cache/<run>/; updates the server did not receive are NaN.
"""
import json
import os

import numpy as np

from attack import dataset


def run_key(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


PROBES = os.path.join(os.path.dirname(__file__), 'probe_states.npy')


def q_values(flat, shapes, S):
    """Q-values (n_states, 3) of the DQN with flattened weights flat (dqn.py: Dense 150 relu, 100 relu, 3)."""
    w, i = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        w.append(flat[i:i + n].reshape(sh).astype(np.float64))
        i += n
    h = np.maximum(S @ w[0] + w[1], 0)
    h = np.maximum(h @ w[2] + w[3], 0)
    return h @ w[4] + w[5]


def build_probe(run_dir, out):
    path = os.path.join(out, 'probe.npy')
    if os.path.exists(path):
        return
    S = np.load(PROBES).astype(np.float64)
    meta = dataset.load_meta(run_dir)
    vehs = [v for v, _ in dataset.load_agents(run_dir)]
    shapes = [tuple(x) for x in meta['layer_shapes']]
    T = meta['rounds']
    probe = np.zeros((T, len(vehs), S.shape[0] * 3), np.float32)
    probe[:] = np.nan
    for t in range(1, T + 1):
        g = dataset.load_global(run_dir, t)
        q0 = q_values(g, shapes, S)
        obs = dataset.load_round(run_dir, t, meta)
        for j, v in enumerate(vehs):
            if v in obs:
                probe[t - 1, j] = (q_values(obs[v], shapes, S) - q0).ravel()
    np.save(path + '.tmp.npy', probe)
    os.replace(path + '.tmp.npy', path)


def build(run_dir, cache_dir, pos_inputs):
    out = os.path.join(cache_dir, run_key(run_dir))
    done = os.path.join(out, 'done.json')
    if os.path.exists(done):
        build_probe(run_dir, out)
        return out
    os.makedirs(out, exist_ok=True)
    meta = dataset.load_meta(run_dir)
    agents = dataset.load_agents(run_dir)
    vehs = [v for v, _ in agents]
    T, V, D = meta['rounds'], len(vehs), meta['parameters']
    units = meta['layer_shapes'][0][1]
    cols = np.concatenate([np.arange(i * units, (i + 1) * units) for i in pos_inputs])
    pos = np.full((T, V, len(cols)), np.nan, np.float32)
    observed = np.zeros((T, V), bool)
    full = np.lib.format.open_memmap(os.path.join(out, 'full.npy'), mode='w+', dtype=np.float16, shape=(T * V, D))
    for t in range(1, T + 1):
        g = dataset.load_global(run_dir, t)
        obs = dataset.load_round(run_dir, t, meta)
        for j, v in enumerate(vehs):
            if v not in obs:          # not received by the server this round (ns-3)
                full[(t - 1) * V + j] = np.nan
                continue
            u = obs[v] - g
            observed[t - 1, j] = True
            pos[t - 1, j] = u[cols]
            full[(t - 1) * V + j] = u.astype(np.float16)
    full.flush()
    del full
    np.save(os.path.join(out, 'pos.npy'), pos)
    np.save(os.path.join(out, 'observed.npy'), observed)
    np.save(os.path.join(out, 'labels.npy'), np.array([l for _, l in agents]))
    with open(done, 'w') as f:
        json.dump({'run_dir': run_dir, 'method': meta['method'], 'eps': meta['eps'], 'seed': meta['seed'],
                   'rounds': T, 'vehicles': V, 'vehs': vehs, 'pos_inputs': list(pos_inputs)}, f, indent=1)
    build_probe(run_dir, out)
    return out


def load(cache_run_dir):
    """Run data with the vehicles in a random order (seeded per run). In the simulation vehicle i starts at origin
    i mod (number of origins), so the index would reveal the origin; the attack sees vehicles only as an unordered set of pseudonymous
    upload histories. 'perm' maps the new order to rows of 'full' (row t * V + perm[j]); info['vehs'] keeps the true
    IDs in the new order, for scoring and per-vehicle output only."""
    with open(os.path.join(cache_run_dir, 'done.json')) as f:
        info = json.load(f)
    labels = np.load(os.path.join(cache_run_dir, 'labels.npy'))
    perm = np.random.default_rng([int(info['seed']), len(labels), 7]).permutation(len(labels))
    info['vehs'] = [info['vehs'][i] for i in perm]
    obs_path = os.path.join(cache_run_dir, 'observed.npy')
    observed = np.load(obs_path) if os.path.exists(obs_path) else np.ones((info['rounds'], len(labels)), bool)
    return {'info': info, 'perm': perm, 'observed': observed[:, perm],
            'pos': np.load(os.path.join(cache_run_dir, 'pos.npy'))[:, perm],
            'full': np.load(os.path.join(cache_run_dir, 'full.npy'), mmap_mode='r'),
            'probe': np.load(os.path.join(cache_run_dir, 'probe.npy'))[:, perm],
            'labels': labels[perm]}
