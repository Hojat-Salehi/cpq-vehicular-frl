# run_client.py
# Unified Flower client for all DP methods: none | cpq | ldpfl | gaussian | laplace
# (cpq / ldpfl: one-bit private quantization of the update, onebit.py)
# Usage: python run_client.py --config config.yaml --vehicle 0

from __future__ import absolute_import, print_function

import os
import logging
os.environ['GRPC_VERBOSITY']      = 'ERROR'
os.environ['GRPC_TRACE']          = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('flwr').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)



import sys
import time
import random
import argparse
import numpy as np

import flwr as fl
import tensorflow as tf
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

from dqnagent import dqnAgent
from shared_env import SharedSumoEnv
from shared_train import (AgentSlot, draw_departures, evaluate as evaluate_episodes,
                          generate_background, greedy_policy, summarize, train_round)
from utils import (
    checkSumo, parseConfig, get_alledges, calculate_connections, save_data,
)
from xml.etree.ElementTree import parse as xml_parse

# ── SUMO must be available (needed for the DQN environment regardless of ns3) ──
if 'SUMO_HOME' not in os.environ:
    sys.exit('Declare environment variable "SUMO_HOME"')

# ── Shared layer name order — must match DQN model construction order ──
# DQN has: fc1 (Dense 150), fc2 (Dense 100), fc_out (Dense action_size)
# Keras get_weights() returns: [fc1.kernel, fc1.bias, fc2.kernel, fc2.bias, fc_out.kernel, fc_out.bias]
LAYER_NAMES = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc_out.weight', 'fc_out.bias']


def get_edgesinfo(net):
    tree = xml_parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    return [x.find("lane").attrib for x in alledgelists]


def generate_lanedetectionfile(net, det):
    alledges   = get_alledges(net)
    edgesinfo  = get_edgesinfo(net)
    alldets    = [edge.replace("E", "D") for edge in alledges]
    with open(det, "w") as f:
        print('<additional>', file=f)
        for i, v in enumerate(edgesinfo):
            print(
                '        <laneAreaDetector id="%s" lane="%s" pos="0.0" '
                'length="%s" freq="%s" file="dqn_detfile.out"/>'
                % (alldets[i], v['id'], v['length'], "1"),
                file=f,
            )
        print('</additional>', file=f)
    return alldets


def get_alldets(alledges):
    return [edge.replace("E", "D") for edge in alledges]


# ─────────────────────────────────────────────────────────────────────────────
class flowerClient(fl.client.NumPyClient):
    """
    Single Flower client that supports all DP methods and optionally the
    ns3/PQC communication channel.
    """

    def __init__(
        self,
        agent,
        trained,
        sumoBinary,
        plotResult,
        num_episode,
        net,
        trip,
        randomrou,
        add,
        dirResult,
        dirModel,
        sumocfg,
        fcdoutput,
        edgelists,
        alldets,
        dict_connection,
        veh,
        destination,
        state_size,
        action_size,
        dqn_config,
        num_seed,
        file_path,
        create_csvs: bool = True,
    ) -> None:
        self.agent          = agent
        self.trained        = trained
        self.num_seed       = num_seed
        self.sumoBinary     = sumoBinary
        self.plotResult     = plotResult
        self.num_episode    = num_episode
        self.net            = net
        self.trip           = trip
        self.randomrou      = randomrou
        self.add            = add
        self.dirResult      = dirResult
        self.dirModel       = dirModel
        self.sumocfg        = sumocfg
        self.fcdoutput      = fcdoutput
        self.edgelists      = edgelists
        self.alldets        = alldets
        self.dict_connection= dict_connection
        self.veh            = veh
        self.destination    = destination
        self.state_size     = state_size
        self.action_size    = action_size
        self.dqn_config     = dqn_config
        self.file_path      = file_path

        # ── Pre-create one CSV per logical client identity ──────────────────
        # When batches_per_agg > 1, each physical client represents multiple
        # logical clients (one per sub-round). We create all their CSV files
        # here at startup so they exist before any fit() call.
        # The shared-environment loop writes its own CSVs (create_csvs=False).
        from utils import setup_csv_stable
        _batches   = int(dqn_config.get('batches_per_agg', 1)) if create_csvs else 0
        _n_clients = int(dqn_config.get('num_clients', 10))
        _out_dir   = dqn_config.get('out_dir', 'results')
        _veh_index = int(veh[3:])          # e.g. 'veh3' → 3
        self._logical_csv_paths = {}       # sub_round_index (0-based) → file path
        for _sub_idx in range(_batches):
            _logical_id  = _sub_idx * _n_clients + _veh_index
            _logical_veh = f'veh{_logical_id}'
            self._logical_csv_paths[_sub_idx] = setup_csv_stable(
                _out_dir, _logical_veh, trained
            )
        self._eval_csv = setup_csv_stable(_out_dir, veh, 'eval') if create_csvs else None
        if create_csvs:
            print(f"[CLIENT {veh}] Created {_batches} logical CSV(s): "
                  f"veh{_veh_index} → "
                  f"{[f'veh{i*_n_clients+_veh_index}' for i in range(_batches)]}")

        # ── Read method and privacy params from config ──
        self.method     = dqn_config.get('method', 'none')
        self.use_ns3    = dqn_config.get('use_ns3', False)
        epsilon         = dqn_config.get('epsilon', 1.0)
        delta           = dqn_config.get('delta', 1e-5)

        # ── Initialise the right DP mechanism ──
        self.dp = None
        self.privacy_epsilon = epsilon
        self.privacy_delta = delta
        if self.method in ('cpq', 'ldpfl'):
            print(f"[CLIENT {self.veh}] DP method: {self.method.upper()}  epsilon={epsilon}  "
                  f"clip_radius={dqn_config.get('clip_radius')}"
                  + (f"  lambda={dqn_config.get('cpq_lambda', 1.0)}" if self.method == 'cpq' else ''))

        elif self.method in ('gaussian', 'laplace'):
            print(f"[CLIENT {self.veh}] DP method: {self.method}  epsilon={epsilon}"
                  + (f"  delta={delta}" if self.method == 'gaussian' else '')
                  + f"  clip_radius={dqn_config.get('clip_radius')}")

        else:
            print(f"[CLIENT {self.veh}] DP method: none (vanilla FedAvg)")

        # ── Initialise ns3 / PQC components only when use_ns3=True ──
        self.crypto = None
        self.oracle  = None
        self._send_pqc_update = None

        if self.use_ns3:
            # These imports trigger require_run_id() at module level — safe now
            # because utils.require_run_id() auto-generates if FL_RUN_ID is unset.
            from pqc_oracle import send_pqc_update as _spu
            from pqc.pqc_crypto import PQCCrypto, verify_and_unpack_packet
            from oracle_transport import OracleTransport

            self._verify_and_unpack = verify_and_unpack_packet
            self._send_pqc_update   = _spu

            pqc_cfg    = dqn_config.get('pqc', {})
            oracle_cfg = dqn_config.get('oracle', {})

            if pqc_cfg.get('enabled', False):
                self.crypto = PQCCrypto(pqc_cfg)
            if oracle_cfg.get('enabled', False):
                self.oracle = OracleTransport(oracle_cfg)
                # Listen before the first round: the server sends round-1 downlinks before fit()
                self.oracle._ensure_downlink_listener(self.veh)

            print(f"[CLIENT {self.veh}] ns3 channel: ENABLED")
        else:
            print(f"[CLIENT {self.veh}] ns3 channel: DISABLED  (pure gRPC)")

    # ─────────────────────────────────────────────────────────────────────────
    def get_parameters(self, config) -> NDArrays:
        return self.agent.model.get_weights()

    # Lets the server map this Flower client to its vehicle id (ns-3 downlink/uplink keys)
    def get_properties(self, config) -> Dict[str, Scalar]:
        return {'veh': self.veh}

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, parameters: NDArrays, config) -> Tuple[NDArrays, int, Dict]:
        t_start = time.time()
        ctx = self.begin_round(parameters, config)

        # ── 4. Local DQN training (SUMO episodes) ───────────────────────────
        # Select the logical CSV for this sub-round (one per logical client
        # identity when batches_per_agg > 1).
        round_num = ctx['round_num']
        _batches_per_agg = max(int(self.dqn_config.get('batches_per_agg', 1)), 1)
        active_file_path = self._logical_csv_paths[(round_num - 1) % _batches_per_agg]

        print(f"[CLIENT {self.veh}] Starting local training... "
              f"(writing to {os.path.basename(active_file_path)})")
        # Same episode loop as the shared environment, with this vehicle alone
        # in its own SUMO (shared_train.py: reward, time limit, logged metrics)
        env, slot = self._episode_env()
        slot.csv_path = active_file_path

        def background(episode):
            if episode % 10 == 0:
                generate_background(self.net, self.add, self.trip, self.randomrou,
                                    seed=self.num_seed * 10000 + ctx['true_round'] * 100 + episode,
                                    period=self.dqn_config['background_period'])
            return self.randomrou

        cfg = self.dqn_config
        train_round(env, [slot], self.num_episode, ctx['true_round'], background, save_data,
                    cfg['max_trip_time'], self._free_flow, None,
                    lambda ep: draw_departures(self._dep_rng, [self.veh], cfg['warmup'],
                                               cfg['depart_window']))
        print(f"[CLIENT {self.veh}] Local training complete")

        send_weights, metrics = self.end_round(ctx)
        fit_ms = (time.time() - t_start) * 1000.0
        print(f"[CLIENT {self.veh}] fit() done in {fit_ms:.0f}ms")
        metrics['fit_ms'] = fit_ms
        return send_weights, 1000, metrics

    def _group_pairing_keys(self):
        """(K_pair, K_coin, own position, group size) from the group secret (cpq_pairing: local).
        Agreed group = all vehicles of the run, ordered by id."""
        if getattr(self, '_pair_keys', None) is None:
            import cpq_pairing
            roster = [f'veh{k}' for k in range(int(self.dqn_config['num_clients']))]
            if self.dqn_config.get('cpq_group_key', 'seeded') == 'seeded':
                secret = cpq_pairing.seeded_group_secret(int(self.dqn_config['seed']))
            else:   # ml_kem: the keys come from the group setup (run_shared.py sets _pair_keys)
                raise ValueError('cpq_group_key ml_kem: the group setup has not given this vehicle its keys '
                                 '(only supported by run_shared.py)')
            kp, kc, _ = cpq_pairing.group_keys(secret, roster)
            self._pair_keys = (kp, kc, roster.index(self.veh), len(roster))
        return self._pair_keys

    def _episode_env(self):
        """This vehicle's SUMO environment and slot (created on first use)."""
        if getattr(self, '_env', None) is None:
            self._env = SharedSumoEnv(
                self.sumoBinary, self.sumocfg, self.edgelists, self.alldets,
                self.dict_connection, self.destination, self.state_size,
                self.action_size, self.dqn_config)
            start = self.dqn_config['start_edges'][int(self.veh[3:])]
            self._free_flow = self._env.free_flow_times([start])
            self._slot = AgentSlot(self.veh, self.agent, start)
            self._dep_rng = np.random.default_rng(self.num_seed * 7919 + 17)
            import atexit
            atexit.register(self._env.close)
        self._slot.agent = self.agent
        return self._env, self._slot

    # ─────────────────────────────────────────────────────────────────────────
    def begin_round(self, parameters: NDArrays, config) -> Dict:
        """Steps before local training: load the global model, reset epsilon, compute C/R.

        Returns the round context that end_round() needs. Used by fit() and by
        the shared-environment loop (run_shared.py), which trains all vehicles together.
        """
        round_num       = int(config.get('round', 0))            # Flower sub-round
        true_round      = int(config.get('true_round', round_num))  # true FL round for epsilon
        true_num_rounds = int(config.get('true_num_rounds',
                            self.dqn_config.get('true_rounds', 50)))
        client_idx      = int(config.get('client_idx', -1))
        pair_seed  = config.get('pair_seed', None)
        role       = int(config.get('role', 1))   # 1=leader, 0=follower (CorBinFL only)

        print(f"\n[CLIENT {self.veh}] ===== Round {true_round} | sub-round {round_num} | method={self.method} =====")
        print(f"[CLIENT {self.veh}] client_idx={client_idx}  pair_seed={pair_seed}  role={role}")

        # ── 1. Resolve incoming weights: ns3 downlink OR gRPC ──────────────
        weights_in = list(parameters)
        dl_ok = 0   # 1 when the global model arrived over the ns-3 downlink

        if self.use_ns3 and self.oracle is not None:
            pkt = None
            try:
                pkt = self.oracle.recv_wave_downlink(self.veh, round_num, timeout=5.0)
            except Exception as e:
                print(f"[CLIENT {self.veh}] Downlink recv error: {e}", flush=True)

            if pkt and self.crypto is not None:
                try:
                    hdr, dl_weights = self._verify_and_unpack(
                        pkt,
                        key_dir=getattr(self.crypto, 'key_dir', 'pqc_keys'),
                    )
                    hveh  = str(hdr.get('veh', ''))
                    hrnd  = int(hdr.get('round', -1))
                    if hrnd == round_num and (hveh == '' or hveh == str(self.veh)):
                        weights_in = dl_weights
                        dl_ok = 1
                        print(
                            f"[CLIENT {self.veh}] Using ns3 downlink weights "
                            f"bytes={len(pkt)} round={round_num}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[CLIENT {self.veh}] Downlink header mismatch "
                            f"(veh={hveh}, rnd={hrnd}); using gRPC params.",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[CLIENT {self.veh}] Downlink unpack failed: {e}; using gRPC.", flush=True)

        # ── 2. Load global weights into agent ───────────────────────────────
        self.agent.model.set_weights(weights_in)
        print(f"[CLIENT {self.veh}] Global model loaded")


        # ── 2b. Per-round epsilon reset (FedRL best practice) ───────────────
        # ── 2b. Per-round epsilon reset (FedRL best practice) ───────────────
        if round_num >= 1:
            _num_rounds  = true_num_rounds
            _eps_start   = float(self.dqn_config.get('epsilon_start', 1.0))
            _eps_end     = float(self.dqn_config.get('epsilon_end', 0.05))
            _eps_sched   = self.dqn_config.get('epsilon_schedule', 'exponential')
            
            # Fetch the new decay limit, defaulting to total rounds if not provided
            _decay_rounds = int(self.dqn_config.get('epsilon_decay_rounds', _num_rounds))

            if _eps_sched == 'linear':
                # Linear: Use _decay_rounds and cap it at _eps_end
                frac          = (true_round - 1) / max(_decay_rounds - 1, 1)
                round_epsilon = max(_eps_start - frac * (_eps_start - _eps_end), _eps_end)
            else:
                # Exponential: Use _decay_rounds
                _decay        = (_eps_end / _eps_start) ** (1.0 / max(_decay_rounds - 1, 1))
                # The max() function acts as a floor. Once true_round > _decay_rounds, 
                # the math drops below _eps_end, and max() forces it to stay flat at _eps_end.
                round_epsilon = max(_eps_start * (_decay ** (true_round - 1)), _eps_end)
                
            self.agent.epsilon = float(round_epsilon)
            print(f"[CLIENT {self.veh}] Epsilon reset to {self.agent.epsilon:.4f} "
                  f"(true_round={true_round}/{_num_rounds}, "
                  f"flower_round={round_num}, schedule={_eps_sched})")

        # ── 3. One-bit methods: the global model is the centre of the clipping interval ──
        self.dqn_config['current_true_round'] = true_round
        return {'round_num': round_num, 'true_round': true_round, 'pair_seed': pair_seed,
                'role': role, 'base': [np.array(w, copy=True) for w in weights_in], 'dl_ok': dl_ok,
                'pairing': config.get('pairing', 'client'), 'pairing_seed': config.get('pairing_seed'),
                'n_clients': config.get('n_clients'), 'client_idx': client_idx}

    # ─────────────────────────────────────────────────────────────────────────
    def end_round(self, ctx: Dict) -> Tuple[NDArrays, Dict]:
        """Steps after local training: privatize the update and (optionally) send it over ns-3."""
        round_num, role = ctx['round_num'], ctx['role']
        trained_weights = self.agent.model.get_weights()
        extra = {}

        # ── 5. Apply DP perturbation ─────────────────────────────────────────
        if self.method in ('cpq', 'ldpfl'):
            import onebit
            from dqn import LAYER_NAMES
            radii = onebit.radii_for(ctx['base'], self.dqn_config['clip_radius'], LAYER_NAMES)
            if self.method == 'cpq' and ctx['pair_seed'] is not None:
                seed = int(ctx['pair_seed'])            # shared with the pair partner
            else:                                       # own, independent randomness
                seed = int(np.random.SeedSequence([int(self.num_seed), int(round_num)]).generate_state(1)[0])
            own_seed = int(np.random.SeedSequence([int(self.num_seed), int(round_num)]).generate_state(1)[0])
            shared_V = None
            if self.method == 'cpq' and self.dqn_config.get('cpq_pairing', 'server') == 'local':
                import cpq_pairing
                kp, kc, index, n = self._group_pairing_keys()
                per_coord = self.dqn_config.get('cpq_pairing_granularity', 'client') == 'coordinate'
                shared_V = cpq_pairing.shared_values(kp, kc, round_num, index, n,
                                                     [w.shape for w in trained_weights], own_seed, per_coord)
            elif self.method == 'cpq' and ctx['pairing'] == 'coordinate':
                import cpq_pairing
                shared_V = cpq_pairing.coordinate_values(int(ctx['pairing_seed']), round_num, int(ctx['client_idx']),
                                                         int(ctx['n_clients']), [w.shape for w in trained_weights],
                                                         own_seed)
            send_weights, clipped = onebit.encode(trained_weights, ctx['base'], radii,
                                                  self.privacy_epsilon, self.method, seed, role,
                                                  cpq_lambda=float(self.dqn_config.get('cpq_lambda', 1.0)),
                                                  private_seed=own_seed, shared_V=shared_V)
            extra = {f'clipped_{name}': c for name, c in zip(LAYER_NAMES, clipped)}
            print(f"[CLIENT {self.veh}] {self.method.upper()} applied (role {role}); clipped share per layer "
                  f"{[round(c, 4) for c in clipped]}")

        elif self.method in ('gaussian', 'laplace'):
            import additive
            from dqn import LAYER_NAMES
            import onebit
            radii = onebit.radii_for(ctx['base'], self.dqn_config['clip_radius'], LAYER_NAMES)
            seed = int(np.random.SeedSequence([int(self.num_seed), int(round_num)]).generate_state(1)[0])
            send_weights, clipped = additive.perturb(trained_weights, ctx['base'], radii, self.privacy_epsilon,
                                                     self.privacy_delta, self.method, seed)
            extra = {f'clipped_{name}': c for name, c in zip(LAYER_NAMES, clipped)}
            print(f"[CLIENT {self.veh}] {self.method} noise applied, scale per layer "
                  f"{[round(additive.noise_scale(self.method, self.privacy_epsilon, self.privacy_delta, r), 5) for r in radii]}; "
                  f"clipped share {[round(c, 4) for c in clipped]}")

        else:  # 'none'
            send_weights = trained_weights
            print(f"[CLIENT {self.veh}] No perturbation (vanilla FedAvg)")

        # ── 6. Save local model ──────────────────────────────────────────────
        self.agent.save_weights()

        # ── 7. ns3 uplink (only when use_ns3=True) ──────────────────────────
        if self.use_ns3 and self._send_pqc_update is not None:
            self._send_pqc_update(
                veh_id=self.veh,
                round_num=round_num,
                weights=send_weights,
                crypto=self.crypto,
                oracle=self.oracle,
            )

        return send_weights, dict({'veh': self.veh, 'dl_ok': ctx['dl_ok']}, **extra)

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self, parameters: NDArrays, config) -> Tuple[float, int, Dict]:
        """Greedy evaluation of the global model on this vehicle's static scenarios."""
        from dqn import DQN
        round_num = int(config.get('round', 0))
        num_eval_episodes = int(config.get('eval_episodes', 3))
        print(f"\n[CLIENT {self.veh}] ===== EVAL Round {round_num} | eps=0 | {num_eval_episodes} scenarios =====")

        if getattr(self, '_eval_model', None) is None:
            self._eval_model = DQN(self.action_size)
            self._eval_model(tf.zeros((1, self.state_size)), training=False)
        self._eval_model.set_weights(list(parameters))

        env, slot = self._episode_env()
        eval_slot = AgentSlot(self.veh, None, slot.start_edges)
        files = [f"Rou/eval_{self.veh}_scenario_{i}.rou.xml" for i in range(num_eval_episodes)]
        # Same fixed departure times per scenario as the shared environment
        cfg = self.dqn_config
        vehs = [f'veh{k}' for k in range(int(cfg.get('num_clients', 1)))]
        departures = lambda j: draw_departures(np.random.default_rng(5000 + j), vehs,
                                               cfg['warmup'], cfg['depart_window'])
        rows = evaluate_episodes(env, [eval_slot], greedy_policy(self._eval_model), files,
                                 cfg['max_trip_time'], self._free_flow, round_num,
                                 save_data, lambda s: self._eval_csv, departures=departures)

        metrics = summarize(rows)
        loss = metrics['tt_sum'] / max(metrics['n'], 1.0)   # mean travel time
        metrics['veh'] = self.veh
        return float(loss), len(rows), metrics

# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    net       = 'Net/dqn.net.xml'
    add       = 'Add/dqn.add.xml'
    det       = 'Add/dqn.det.xml'
    sumocfg   = 'dqn.sumocfg'
    dirResult = 'Result/dqn'
    dirModel  = 'Avg_Saved_Models/'
    fcdoutput = 'Output/dqn.fcdoutput.xml'
    destination = 'E15'
    action_size = 3
    trained     = False

    parser = argparse.ArgumentParser(description='Unified FLDQN Client')
    parser.add_argument('--config',  type=str, default='configs/base.yaml')
    parser.add_argument('--vehicle', type=int, required=True)
    args = parser.parse_args()

    traci, sumolib = checkSumo()
    config         = parseConfig(args)

    veh_index  = int(args.vehicle)
    veh_id     = f'veh{veh_index}'
    trip       = f'Rou/dqn_{veh_index}.trip.xml'
    randomrou  = f'Rou/dqnrandom_{veh_index}.rou.xml'

    edgelists      = get_alledges(net)
    state_size     = SharedSumoEnv.state_dim(config['state_encoding'], len(edgelists))
    dict_connection = calculate_connections(edgelists, net)
    dets           = generate_lanedetectionfile(net, det)
    alldets        = get_alldets(edgelists)

    # Training CSVs (one per logical client) are created inside flowerClient.__init__
    file_path = None

    if config['use_gui']:
        sumoBinary = sumolib.checkBinary('sumo-gui')
    else:
        sumoBinary = sumolib.checkBinary('sumo')

    num_episode = config['total_episodes']
    num_seed    = int(config['seed']) * 1000 + veh_index   # per-vehicle, reproducible per run seed

    agent = dqnAgent(
        veh_id, num_seed, edgelists, dict_connection,
        state_size, action_size, num_episode, dirModel,
        target_update_every=config['target_update_every'],
        mask_target=config['mask_target'],
    )

    agent.model.build(input_shape=(None, state_size))

    client = flowerClient(
        agent=agent,
        trained=trained,
        sumoBinary=sumoBinary,
        plotResult=config['plot_results'],
        num_episode=num_episode,
        net=net,
        trip=trip,
        randomrou=randomrou,
        add=add,
        dirResult=dirResult,
        dirModel=dirModel,
        sumocfg=sumocfg,
        fcdoutput=fcdoutput,
        edgelists=edgelists,
        alldets=alldets,
        dict_connection=dict_connection,
        veh=veh_id,
        destination=destination,
        state_size=state_size,
        action_size=action_size,
        dqn_config=config,
        num_seed=num_seed,
        file_path=file_path,
    )

    server_address = f"{config['server_ip']}:{config['port']}"
    print(f'[CLIENT {veh_id}] Connecting to {server_address}')
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )