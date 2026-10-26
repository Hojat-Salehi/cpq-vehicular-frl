# attack_log.py
# What the server observes, logged per round for the origin inference attack (attack/,
# Attack_Exp/destination_inference_attack_spec.md, target changed to the origin). Switched on by
# FL_ATTACK_LOG=1 (tools/run_experiment.py sets it for experiment ATK). Logging only: it saves
# arrays run_shared.py already has and draws no random numbers, so the run is unchanged.
#
# <out_dir>/attack_log/
#   meta.json               method, eps, seed, vehicles, layer names and shapes, origins, start / end
#   agents.csv              veh, origin edge, origin label (index in the distinct sumo.start_edges),
#                           origin x, y (start of lane 0 of the origin edge, as in the state)
#   global/round_XXX.npy    flattened float32 global model sent to the vehicles in round XXX
#   uploads/round_XXX.npz   every vehicle's update exactly as the server received it (the
#                           FitRes parameters), key <veh>__<k> = k-th array:
#                           cpq / ldpfl: the wire format (packed bits of layer 0, shape of layer 0, ...)
#                           none / gaussian / laplace: float32 layer k

import datetime
import json
import os

import numpy as np


def enabled():
    return os.environ.get('FL_ATTACK_LOG') == '1'


class AttackLog:
    def __init__(self, out_dir, cfg, slots, layer_names, net_file):
        import sumolib
        self.dir = os.path.join(out_dir, 'attack_log')
        for sub in ('global', 'uploads'):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        net = sumolib.net.readNet(net_file)
        origins = list(dict.fromkeys(s.start_edges[0] for s in slots))
        coords = {e: tuple(net.getEdge(e).getLane(0).getShape()[0]) for e in origins}
        with open(os.path.join(self.dir, 'agents.csv'), 'w') as f:
            f.write('veh_id,origin_id,origin_label,origin_x,origin_y\n')
            for s in slots:
                e = s.start_edges[0]
                f.write(f'{s.veh},{e},{origins.index(e)},{coords[e][0]:g},{coords[e][1]:g}\n')
        shapes = [list(w.shape) for w in slots[0].agent.model.get_weights()]
        self.meta = {
            'method': cfg['method'], 'eps': None if cfg['method'] == 'none' else float(cfg['epsilon']),
            'seed': cfg['seed'], 'rounds': cfg['true_rounds'], 'vehicles': len(slots),
            'clip_radius': cfg['clip_radius'], 'cpq_lambda': cfg.get('cpq_lambda'), 'use_ns3': cfg['use_ns3'],
            'layer_names': list(layer_names), 'layer_shapes': shapes,
            'parameters': int(sum(np.prod(s) for s in shapes)),
            'origins': [{'label': i, 'edge': e, 'x': coords[e][0], 'y': coords[e][1]} for i, e in enumerate(origins)],
            'start': datetime.datetime.now().isoformat(timespec='seconds'), 'end': None, 'rounds_logged': 0,
        }
        self._write_meta()

    def _write_meta(self):
        with open(os.path.join(self.dir, 'meta.json'), 'w') as f:
            json.dump(self.meta, f, indent=1)

    def round(self, rnd, global_weights, received):
        """received: {veh: list of arrays as the server got them}."""
        flat = np.concatenate([np.asarray(w, np.float32).ravel() for w in global_weights])
        np.save(os.path.join(self.dir, 'global', f'round_{rnd:03d}.npy'), flat)
        arrays = {f'{veh}__{k}': np.asarray(a) for veh, arrs in received.items() for k, a in enumerate(arrs)}
        np.savez(os.path.join(self.dir, 'uploads', f'round_{rnd:03d}.npz'), **arrays)
        self.meta['rounds_logged'] = rnd

    def close(self):
        self.meta['end'] = datetime.datetime.now().isoformat(timespec='seconds')
        self._write_meta()
