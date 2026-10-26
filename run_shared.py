# run_shared.py
# Federated DQN with all vehicles in ONE shared SUMO world (design change 1).
#
# Replaces "one run_server.py + one run_client.py process per vehicle" with one
# process. Per round:
#   1. the strategy (same classes as run_server.py) configures every vehicle
#      (pairing / roles for CPQ, true_round for the epsilon schedule)
#   2. each vehicle's flowerClient.begin_round() loads the global model
#   3. all vehicles train together for total_episodes shared episodes
#   4. each vehicle's flowerClient.end_round() privatizes its update
#   5. strategy.aggregate_fit() builds the new global model
#   6. the global model is evaluated greedily, all vehicles in the shared world
#
# With ns3.use_ns3: true the model exchange is simulated (comm.py): the global model is
# delivered during a downlink drive before training and the updates during an uplink drive
# after it, over ns-3 802.11p with PQC signatures, from the vehicles' SUMO positions to one
# RSU. Only verified updates from vehicles that trained from the current global model are
# aggregated; a vehicle that missed the downlink keeps training its own previous model.
# With use_ns3: false the exchange is ideal (instant, every vehicle participates).
#
# Agents depart after a warm-up of sumo.warmup seconds, at random times within
# sumo.depart_window (re-drawn every training episode, fixed per evaluation
# scenario). Before round 1 the evaluation scenarios are also run with two
# non-learning baselines (shortest route, re-routing on current travel times)
# and without agents.
#
# Outputs (out_dir): vehK_train.csv / vehK_eval.csv (one row per vehicle per
# episode, see shared_train.CSV_HEADER), global_eval.csv, round_times.csv,
# baselines.csv + baseline_eval.csv (same columns, per baseline policy),
# background.csv (non-agent traffic in each evaluation scenario over the fixed
# window [warmup, warmup + depart_window + max_trip_time], with
# experiment.background_metrics) and
# road_log.csv (per vehicle per edge, with experiment.road_log).
#
# Usage: python run_shared.py --config configs/base.yaml
# FL_SEED / FL_OUT_DIR / FL_RUN_ID come from the environment (tools/run_experiment.py sets them).

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import argparse
import random
import time

import numpy as np
import flwr as fl
import tensorflow as tf
from flwr.common import Code, FitRes, GetPropertiesRes, Status
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from frl.learning.dqn import DQN
from frl.learning.dqnagent import dqnAgent
from frl.federated.run_client import flowerClient, generate_lanedetectionfile, get_alldets
from frl.federated.run_server import build_strategy, get_eval_fn, make_fit_config_fn
from frl.env.shared_env import SharedSumoEnv
from frl.env.shared_train import (CSV_HEADER, EVAL_SUMMARY_HEADER, ROAD_HEADER, AgentSlot,
                          background_csv, draw_departures, evaluate, generate_background,
                          generate_eval_scenario, greedy_policy, router_policy, run_episode,
                          summarize, summary_row, train_round)
from frl.utils import (calculate_connections, checkSumo, get_alledges, parseConfig,
                   require_run_id, save_data)

NET, ADD, DET, SUMOCFG = 'sumo/net/dqn.net.xml', 'sumo/add/dqn.add.xml', 'sumo/add/dqn.det.xml', 'sumo/dqn.sumocfg'
DIR_RESULT, FCD = 'Result/dqn', 'Output/dqn.fcdoutput.xml'
DESTINATION, ACTION_SIZE = 'E15', 3


class LocalClientProxy(ClientProxy):
    """Stands in for a remote Flower client so the strategies run unchanged in-process."""

    def __init__(self, cid, veh):
        super().__init__(cid)
        self.veh = veh

    def get_properties(self, ins, timeout, group_id):
        return GetPropertiesRes(status=Status(Code.OK, ''), properties={'veh': self.veh})

    def get_parameters(self, ins, timeout, group_id):
        raise NotImplementedError

    def fit(self, ins, timeout, group_id):
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


LAYER_NAMES = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc_out.weight', 'fc_out.bias']
UPDATE_QUANTILES = [50, 90, 99, 99.5, 99.9, 100]
UPDATE_STATS_HEADER = (['round', 'layer', 'vehicles', 'from_current_global', 'mean_abs', 'rms']
                       + [f'p{q:g}' for q in UPDATE_QUANTILES] + ['vehicle_p99_median', 'vehicle_p99_max'])
GLOBAL_WEIGHTS_HEADER = ['round', 'layer', 'rms', 'mean_abs', 'max_abs', 'half_range', 'finite']


def write_update_stats(path, rnd, global_weights, local):
    """Per layer: distribution of |w_i - w_base| over all vehicles' coordinates, where w_base is
    the model the vehicle started the round from (the global model, or its own previous model
    when it missed the downlink). Measurement only."""
    current = sum(all(np.array_equal(b, g) for b, g in zip(base, global_weights)) for _, base in local.values())
    with open(path, 'a') as f:
        for k, name in enumerate(LAYER_NAMES):
            per_vehicle = [np.abs(w[k] - base[k]).ravel() for w, base in local.values()]
            d = np.concatenate(per_vehicle)
            q = np.percentile(d, UPDATE_QUANTILES)
            vp99 = [np.percentile(x, 99) for x in per_vehicle]
            row = [rnd, name, len(per_vehicle), current, d.mean(), np.sqrt((d ** 2).mean())] + list(q) \
                + [float(np.median(vp99)), float(np.max(vp99))]
            f.write(','.join(f'{x:.6g}' if isinstance(x, float) else str(x) for x in row) + '\n')


def write_global_weights(path, rnd, weights):
    """Per layer size of the global model after aggregation (divergence check). Measurement only."""
    with open(path, 'a') as f:
        for name, w in zip(LAYER_NAMES, weights):
            w = np.asarray(w, dtype=np.float64)
            fin = np.isfinite(w).all()
            row = [rnd, name, np.sqrt((w ** 2).mean()), np.abs(w).mean(), np.abs(w).max(),
                   (w.max() - w.min()) / 2, int(fin)]
            f.write(','.join(f'{x:.6g}' if isinstance(x, float) else str(x) for x in row) + '\n')


def _new_csv(path, header):
    with open(path, 'w', newline='') as f:
        f.write(','.join(header) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description='FLDQN, shared SUMO environment')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    args.vehicle = None
    cfg = parseConfig(args)

    if cfg['batches_per_agg'] != 1:
        raise SystemExit('run_shared.py: batches_per_agg must be 1 (every vehicle is a real '
                         'agent in the shared world; raise sumo.num_clients instead)')

    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    method, n = cfg['method'], cfg['num_clients']
    n_rounds, num_episode = cfg['true_rounds'], cfg['total_episodes']
    out_dir = os.environ.get('FL_OUT_DIR') or os.path.join(
        'experiments', f"shared_{method}_nc{n}_ep{cfg['eval_episodes']}"
                       f"_eps{cfg['epsilon']}_seed{seed}")
    cfg['out_dir'] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))   # the run documents itself
    # Training traffic lives in the run's own folder, so parallel runs cannot overwrite it
    background_rou = os.path.join(out_dir, 'background_train.rou.xml')
    background_trip = os.path.join(out_dir, 'background_train.trip.xml')
    dir_model = os.path.join(out_dir, 'models') + os.sep     # local models, per run
    run_id = require_run_id()
    print(f'[SHARED] run_id={run_id} method={method} vehicles={n} rounds={n_rounds} '
          f'episodes/round={num_episode} seed={seed} out={out_dir}', flush=True)

    traci, sumolib = checkSumo()
    sumo_binary = sumolib.checkBinary('sumo-gui' if cfg['use_gui'] else 'sumo')
    edgelists = get_alledges(NET)
    dict_connection = calculate_connections(edgelists, NET)
    generate_lanedetectionfile(NET, DET)
    alldets = get_alldets(edgelists)
    STATE_SIZE = SharedSumoEnv.state_dim(cfg['state_encoding'], len(edgelists))

    # ── Vehicles: one DQN agent + client-side FL logic each ──────────────────
    slots, clients, proxies = [], {}, {}
    cm = SimpleClientManager()
    for i in range(n):
        veh = f'veh{i}'
        agent = dqnAgent(veh, seed * 1000 + i, edgelists, dict_connection,
                         STATE_SIZE, ACTION_SIZE, num_episode, dir_model,
                         target_update_every=cfg['target_update_every'], mask_target=cfg['mask_target'])
        # use_ns3=False: the client must not open its own oracle/ns-3 path (comm.py does it)
        clients[veh] = flowerClient(
            agent=agent, trained=False, sumoBinary=sumo_binary,
            plotResult=False, num_episode=num_episode, net=NET,
            trip=None, randomrou=None, add=ADD, dirResult=DIR_RESULT,
            dirModel=dir_model, sumocfg=SUMOCFG, fcdoutput=FCD,
            edgelists=edgelists, alldets=alldets, dict_connection=dict_connection,
            veh=veh, destination=DESTINATION, state_size=STATE_SIZE,
            action_size=ACTION_SIZE, dqn_config=dict(cfg, use_ns3=False), num_seed=seed * 1000 + i,
            file_path=None, create_csvs=False,
        )
        csv_path = _new_csv(os.path.join(out_dir, f'{veh}_train.csv'), CSV_HEADER)
        slots.append(AgentSlot(veh, agent, cfg['start_edges'][i], csv_path))
        proxies[veh] = LocalClientProxy(str(i), veh)
        cm.register(proxies[veh])

    # The strategies' own ns-3 oracle path (separate processes) is not used here
    os.environ['FL_PQC_ENABLED'] = os.environ['FL_ORACLE_ENABLED'] = '0'
    # Origin inference attack (attack_log.py): log what the server receives, with FL_ATTACK_LOG=1
    from frl.monitoring import attack_log
    alog = attack_log.AttackLog(out_dir, cfg, slots, LAYER_NAMES, NET) if attack_log.enabled() else None
    # Aggregation error of the privacy mechanism (mse_log.py), with FL_MSE_LOG=1
    from frl.monitoring import mse_log
    from frl.privacy import onebit
    mlog = (mse_log.MseLog(out_dir, LAYER_NAMES, lambda b: onebit.radii_for(b, cfg['clip_radius'], LAYER_NAMES))
            if mse_log.enabled() and method != 'none' else None)

    strategy = build_strategy(
        cfg,
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=n, min_evaluate_clients=0, min_available_clients=n,
        on_fit_config_fn=make_fit_config_fn(1, n_rounds),
        evaluate_fn=get_eval_fn(),
    )

    env = SharedSumoEnv(sumo_binary, SUMOCFG, edgelists, alldets, dict_connection,
                        DESTINATION, STATE_SIZE, ACTION_SIZE, cfg)
    t_max = cfg['max_trip_time']
    free_flow = env.free_flow_times(cfg['start_edges'])
    warmup, window = cfg['warmup'], cfg['depart_window']
    horizon = warmup + window + t_max                       # fixed evaluation window end
    vehs = [s.veh for s in slots]
    print(f"[SHARED] state encoding {cfg['state_encoding']} ({STATE_SIZE} inputs)", flush=True)
    print(f"[SHARED] background: 1 vehicle every {cfg['background_period']:g}s; agents depart in "
          f"[{warmup:g}, {warmup + window:g}]s; trip time limit {t_max:.0f}s; "
          f"free-flow times {free_flow}", flush=True)
    train_dep_rng = np.random.default_rng(seed * 7919 + 17)

    comm, versions, part_csv = None, {v: 0 for v in vehs}, None
    if cfg['use_ns3']:
        import sumolib as _sumolib
        from frl.comm.comm import CommManager
        rsu = _sumolib.net.readNet(NET).getNode(cfg['ns3']['rsu_junction']).getCoord()
        comm = CommManager(cfg, vehs, out_dir, rsu)
        part_csv = _new_csv(os.path.join(out_dir, 'participation.csv'),
                            ['round', 'downlink_sent', 'downlink_delivered', 'outdated_vehicles',
                             'uplink_sent', 'uplink_delivered', 'aggregated'])
        rule = (f"beacons ({comm.beacon_min} of those in {comm.beacon_timeout:g} s)" if comm.discovery == 'beacon'
                else f"distance <= {comm.radius:g} m")
        print(f"[SHARED] ns-3 communication: RSU at {cfg['ns3']['rsu_junction']} {rsu}, in range by {rule}, "
              f"{cfg['pqc']['scheme']}", flush=True)
    eval_departures = [draw_departures(np.random.default_rng(5000 + j), vehs, warmup, window)
                       for j in range(cfg['eval_episodes'])]

    road_log, tripinfo_dir = None, None
    if cfg['road_log']:
        road_log = _new_csv(os.path.join(out_dir, 'road_log.csv'), ROAD_HEADER)
    if cfg['background_metrics']:
        tripinfo_dir = os.path.join(out_dir, 'tripinfo')
        os.makedirs(tripinfo_dir, exist_ok=True)

    eval_files = []
    for j in range(cfg['eval_episodes']):
        path = f"sumo/rou/eval_shared_p{cfg['background_period']:g}_scenario_{j}.rou.xml"
        if not os.path.exists(path):
            generate_eval_scenario(NET, path, seed=1000 + j, period=cfg['background_period'])
        eval_files.append(path)
    eval_model = DQN(ACTION_SIZE)
    eval_model(tf.zeros((1, STATE_SIZE)), training=False)

    eval_csvs = {s.veh: _new_csv(os.path.join(out_dir, f'{s.veh}_eval.csv'), CSV_HEADER)
                 for s in slots}
    summary_csv = _new_csv(os.path.join(out_dir, 'global_eval.csv'), EVAL_SUMMARY_HEADER)
    # Measurement only: size of the vehicles' local updates and of the global model per layer
    update_csv = _new_csv(os.path.join(out_dir, 'update_stats.csv'), UPDATE_STATS_HEADER)
    global_csv = _new_csv(os.path.join(out_dir, 'global_weights.csv'), GLOBAL_WEIGHTS_HEADER)
    times_csv = _new_csv(os.path.join(out_dir, 'round_times.csv'),
                         ['round', 'train_s', 'aggregate_s', 'eval_s', 'round_s', 'num_updates'])

    def tripinfo_for(label):
        if not tripinfo_dir:
            return None
        return lambda j: os.path.join(tripinfo_dir, f'eval_r{label}_s{j}.xml')

    # Reference points on the evaluation scenarios: no agents (background only)
    # and the two non-learning baselines controlling the agent vehicles
    if tripinfo_dir:
        for j, path in enumerate(eval_files):
            run_episode(env, [], path, None, False, t_max, tripinfo_for(0)(j), None, horizon)
    base_rows_csv = _new_csv(os.path.join(out_dir, 'baseline_eval.csv'), CSV_HEADER)
    base_csv = _new_csv(os.path.join(out_dir, 'baselines.csv'), EVAL_SUMMARY_HEADER)
    for mode in ('shortest', 'reroute'):
        rows = evaluate(env, slots, router_policy(env, mode), eval_files, t_max, free_flow, mode,
                        save_data, lambda s: base_rows_csv, None, tripinfo_for(mode),
                        lambda j: eval_departures[j], horizon)
        summary = summary_row(mode, [summarize(rows)])
        save_data(base_csv, summary)
        print(f'[SHARED] Baseline {mode:8s} | travel mean={summary[1]:.1f}s std={summary[2]:.1f}s '
              f'| arrived={summary[3]:.0%} | time loss={summary[4]:.1f}s', flush=True)

    if method == 'cpq' and cfg['cpq_pairing'] == 'local' and cfg['cpq_group_key'] == 'ml_kem':
        # Group setup (paper: ML-KEM key distribution + key confirmation) once before round 1;
        # every vehicle gets the keys it derived itself from K_G
        from frl.privacy import cpq_pairing
        from frl.privacy import group_setup
        roster = [s.veh for s in slots]
        secret = cpq_pairing.seeded_group_secret(seed)
        if comm:
            # Over ns-3 (comm.csv, FL_Round 0, Direction 'setup'): setup drives until every vehicle
            # verified SETUP_OK; the RSU relays the messages
            setup_vehicles = {v: group_setup.Vehicle(v, roster, cfg['pqc']['scheme'], cfg['pqc']['key_dir'])
                              for v in roster}

            def setup_route(d):
                generate_background(NET, ADD, background_trip, background_rou, seed=seed * 100000 + 90 + d,
                                    period=cfg['background_period'])
                return background_rou
            drives = comm.run_setup(env, slots, setup_route,
                                    lambda d: draw_departures(train_dep_rng, vehs, warmup, window),
                                    t_max, setup_vehicles, secret, log=lambda m: print(m, flush=True))
            keys = {v: setup_vehicles[v].keys for v in roster}
            print(f'[SETUP] completed over ns-3 in {drives} drive(s)', flush=True)
        else:
            keys, records = group_setup.run_in_process(
                roster, cfg['pqc']['scheme'], cfg['pqc']['key_dir'], secret, log=lambda m: print(m, flush=True))
            setup_csv = _new_csv(os.path.join(out_dir, 'group_setup.csv'), ['type', 'from', 'to', 'bytes'])
            for rec in records:
                save_data(setup_csv, list(rec))
        for i, v in enumerate(roster):
            clients[v]._pair_keys = (keys[v][0], keys[v][1], i, len(roster))

    # Flower asks one client for the initial model when the strategy has none
    params = strategy.initialize_parameters(cm) or fl.common.ndarrays_to_parameters(
        slots[0].agent.model.get_weights())

    t_run = time.time()
    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        fit_ins = strategy.configure_fit(rnd, params, cm)

        def background(episode, rnd=rnd):
            if episode % 10 == 0:
                generate_background(NET, ADD, background_trip, background_rou,
                                    seed=seed * 100000 + rnd * 100 + episode,
                                    period=cfg['background_period'])
            return background_rou

        background(0)                     # this round's traffic (also used by the drives)
        global_weights = fl.common.parameters_to_ndarrays(params)
        if comm:
            dl_packets = {v: comm.packet(v, rnd, global_weights, 'downlink')
                          for v in vehs if versions[v] < rnd}
            received = comm.run_drive(env, slots, background_rou,
                                      draw_departures(train_dep_rng, vehs, warmup, window),
                                      t_max, 'downlink', rnd, dl_packets)
            for v in received:
                versions[v] = rnd
        ctxs, base_weights = {}, {}
        for proxy, ins in fit_ins:
            config = dict(ins.config)
            config.setdefault('round', rnd)
            config['true_round'], config['true_num_rounds'] = rnd, n_rounds
            if comm and versions[proxy.veh] < rnd:   # missed the downlink: own previous model
                weights_in = clients[proxy.veh].agent.model.get_weights()
            else:
                weights_in = received[proxy.veh] if comm else fl.common.parameters_to_ndarrays(ins.parameters)
            base_weights[proxy.veh] = [np.array(w, copy=True) for w in weights_in]
            ctxs[proxy.veh] = clients[proxy.veh].begin_round(weights_in, config)

        train_round(env, slots, num_episode, rnd, lambda ep: background(ep) if ep else background_rou,
                    save_data, t_max, free_flow, road_log,
                    lambda ep: draw_departures(train_dep_rng, vehs, warmup, window))
        t1 = time.time()

        write_update_stats(update_csv, rnd, global_weights,
                           {v: (clients[v].agent.model.get_weights(), base_weights[v]) for v in base_weights})
        if mlog:
            mlog.updates({v: (clients[v].agent.model.get_weights(), base_weights[v]) for v in base_weights})
        updates = {proxy.veh: clients[proxy.veh].end_round(ctxs[proxy.veh]) for proxy, _ in fit_ins}
        if comm:
            # Only vehicles that trained from the current global model upload: the server
            # decodes (CPQ/LDP-FL) and averages relative to that model
            ul_packets = {v: comm.packet(v, rnd, updates[v][0], 'uplink')
                          for v in updates if versions[v] == rnd}
            arrived = comm.run_drive(env, slots, background_rou,
                                     draw_departures(train_dep_rng, vehs, warmup, window),
                                     t_max, 'uplink', rnd, ul_packets)
            updates = {v: (arrived[v], updates[v][1]) for v in arrived}
        results = [(proxy, FitRes(status=Status(Code.OK, ''),
                                  parameters=fl.common.ndarrays_to_parameters(updates[proxy.veh][0]),
                                  num_examples=1000, metrics=updates[proxy.veh][1]))
                   for proxy, _ in fit_ins if proxy.veh in updates]
        if comm:
            save_data(part_csv, [rnd, len(dl_packets), len(received),
                                 sum(versions[v] < rnd for v in vehs), len(ul_packets),
                                 len(arrived), len(results)])
            print(f'[SHARED] Round {rnd} communication | downlink {len(received)}/{len(dl_packets)} '
                  f'| uplink {len(arrived)}/{len(ul_packets)} | aggregated {len(results)}/{n}',
                  flush=True)
        if alog:
            alog.round(rnd, global_weights, {p.veh: fl.common.parameters_to_ndarrays(r.parameters) for p, r in results})
        new_params, _ = (strategy.aggregate_fit(rnd, results, []) if results else (None, {}))
        if new_params is not None:
            params = new_params
        if mlog and new_params is not None:
            mlog.round(rnd, [p.veh for p, _ in results], fl.common.parameters_to_ndarrays(new_params))
        t2 = time.time()

        write_global_weights(global_csv, rnd, fl.common.parameters_to_ndarrays(params))
        eval_model.set_weights(fl.common.parameters_to_ndarrays(params))
        rows = evaluate(env, slots, greedy_policy(eval_model), eval_files, t_max, free_flow, rnd,
                        save_data, lambda s: eval_csvs[s.veh], road_log, tripinfo_for(rnd),
                        lambda j: eval_departures[j], horizon if tripinfo_dir else None)
        summary = summary_row(rnd, [summarize(rows)])
        save_data(summary_csv, summary)
        t3 = time.time()

        save_data(times_csv, [rnd, round(t1 - t0, 2), round(t2 - t1, 2),
                              round(t3 - t2, 2), round(t3 - t0, 2), len(results)])
        print(f'[SHARED] Round {rnd} GLOBAL EVAL | travel mean={summary[1]:.1f}s '
              f'std={summary[2]:.1f}s | arrived={summary[3]:.0%} | time loss={summary[4]:.1f}s | '
              f'train={t1 - t0:.1f}s eval={t3 - t2:.1f}s', flush=True)

    env.close()
    if alog:
        alog.close()
    if comm:
        comm.close()
    for path in (background_rou, background_trip, background_rou.replace('.rou.xml', '.rou.alt.xml')):
        if os.path.exists(path):
            os.remove(path)
    if tripinfo_dir:
        background_csv(tripinfo_dir, os.path.join(out_dir, 'background.csv'),
                       set(vehs), window_start=warmup)
    print(f'[SHARED] done in {time.time() - t_run:.1f}s; results in {out_dir}', flush=True)


if __name__ == '__main__':
    main()
