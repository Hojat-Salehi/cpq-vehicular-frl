# Server side of CPQ, CPQ-lambda and LDP-FL: each vehicle's bits are decoded relative to the
# round's global model and the decoded models are averaged.

import time
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np

from frl.privacy import onebit
from frl.learning.dqn import LAYER_NAMES
from frl.federated.main_dp_func import federated_learning_pairing, get_seed_for_client
from frl.federated.server import PqcFedAvg, _wait_for_pqc_npz
from frl.utils import client_veh_id, ns3_drop_undelivered


class OneBitStrategy(PqcFedAvg):
    def __init__(self, method, epsilon, clip_radius, lambda_param=1.0, *args,
                 pairing_granularity='client', pairing_seed=0, pairing_mode='server', **kwargs):
        super().__init__(*args, **kwargs)
        if pairing_granularity not in ('client', 'coordinate'):
            raise ValueError(f'cpq_pairing_granularity must be client or coordinate, not {pairing_granularity!r}')
        self.pairing_granularity = pairing_granularity
        if pairing_mode not in ('server', 'local'):
            raise ValueError(f'cpq_pairing must be server or local, not {pairing_mode!r}')
        self.pairing_mode = pairing_mode
        self.pairing_seed = int(pairing_seed)
        if method not in onebit.METHODS:
            raise ValueError(f'OneBitStrategy: unknown method {method!r}')
        self.method = method
        self.epsilon = float(epsilon)
        self.clip_radius = clip_radius
        self.lambda_param = float(lambda_param)
        self.current_global = None
        print(f'[SERVER] {method.upper()}: epsilon={self.epsilon} alpha={onebit.alpha(self.epsilon):.4f} '
              f'clip_radius={clip_radius} lambda_smooth={self.lambda_param}'
              + (f' pairing={pairing_mode}/{pairing_granularity}' if method == 'cpq' else ''), flush=True)

    def configure_fit(self, server_round, parameters, client_manager):
        # PqcFedAvg: sampling, on_fit_config_fn, and the ns-3 downlink in the old layout
        base = super().configure_fit(server_round, parameters, client_manager)
        self.current_global = fl.common.parameters_to_ndarrays(parameters)
        clients = sorted((c for c, _ in base), key=lambda c: int(c.cid) if str(c.cid).isdigit() else str(c.cid))
        config_of = {c.cid: ins.config for c, ins in base}

        pairs, seeds, roles = [], [], {}
        if (self.method == 'cpq' and self.pairing_mode == 'server' and self.pairing_granularity == 'client'
                and len(clients) >= 2):
            pairs, seeds = federated_learning_pairing(len(clients), random_seed=server_round * 42)
            rng = np.random.default_rng(server_round * 123)
            for a, b in pairs:          # leader / follower by coin flip (as in the previous code;
                lead_a = rng.random() < 0.5     # with an odd count the extra pair re-assigns one role)
                roles[a], roles[b] = (1, 0) if lead_a else (0, 1)

        out = []
        for idx, client in enumerate(clients):
            config = dict(config_of[client.cid])      # FedAvg shares one config dict: copy
            config.update(round=server_round, client_idx=idx)
            if self.method == 'cpq' and self.pairing_mode == 'local':
                pass    # the vehicles derive pairing and shared values from their group secret
            elif self.method == 'cpq' and self.pairing_granularity == 'coordinate':
                # every client derives the per-coordinate pairing itself (cpq_pairing.py) from this
                # seed, the round and its position idx in the sorted list of n clients
                config.update(pairing='coordinate', pairing_seed=self.pairing_seed, n_clients=len(clients))
            elif self.method == 'cpq':
                config['pair_seed'] = get_seed_for_client(idx, pairs, seeds) if pairs else None
                config['role'] = roles.get(idx, 1)
            out.append((client, fl.common.FitIns(parameters, config)))
        return out

    def aggregate_fit(self, server_round, results, failures):
        if not results or self.current_global is None:
            return None, {}
        g = self.current_global
        radii = onebit.radii_for(g, self.clip_radius, LAYER_NAMES)
        oracle = self.crypto is not None and self.oracle is not None
        decoded, missing = [], 0
        for idx, (client, fit_res) in enumerate(results):
            packed = None
            if oracle:      # old layout with ns-3: use the update that arrived over the channel
                veh = self._cid_to_veh.get(str(client.cid)) or (fit_res.metrics or {}).get('veh')
                packed = _wait_for_pqc_npz(str(veh), server_round, timeout_s=1.0)
                if packed is None:
                    missing += 1
                    if ns3_drop_undelivered():
                        continue
            if packed is None:
                packed = fl.common.parameters_to_ndarrays(fit_res.parameters)
            decoded.append(onebit.decode(packed, g, radii, self.epsilon))
        if not decoded:
            print('[SERVER] No delivered updates this round; global model unchanged', flush=True)
            return None, {'missing': float(missing)}

        avg = [np.mean([d[k] for d in decoded], axis=0) for k in range(len(g))]
        if self.lambda_param < 1.0:
            avg = [(1.0 - self.lambda_param) * w0 + self.lambda_param * w1 for w0, w1 in zip(g, avg)]
        print(f'[SERVER] {self.method.upper()} round {server_round}: decoded and averaged '
              f'{len(decoded)} updates (missing {missing})', flush=True)
        return fl.common.ndarrays_to_parameters([np.asarray(w, np.float32) for w in avg]), \
            {'aggregated': float(len(decoded)), 'missing': float(missing)}
