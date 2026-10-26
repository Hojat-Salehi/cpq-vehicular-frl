# tests/check_shared_env.py
# Correctness checks for the episode loop (shared_train.py) and SharedSumoEnv.
#   python tests/check_shared_env.py        (inside sumo_env, from the repo root)
#
# 1. state cache   the per-step cached state equals dqnEnv.get_state
# 2. transitions   with 4 learning agents: every chosen edge is a real connection,
#                  next_state of a transition is the state at that agent's next
#                  decision, each agent stores only its own transitions, gradient
#                  steps happen, agents share roads; the final transition is
#                  terminal on arrival and non-terminal (bootstrap) on the time limit
# 3. reward/time   rewards without penalties sum to -Travel_Time exactly; nobody
#                  exceeds the time limit
# 4. metrics       route length, waiting time, time loss, speed are consistent with
#                  each other and with SUMO's own tripinfo; free-flow time is a
#                  lower bound of the agents' travel times; per-road rows add up
#                  departures happen at/after the scheduled time, the time limit
#                  counts from it
# 5. baselines     the shortest-route baseline reproduces the free-flow time in an
#                  empty network; both baselines arrive with traffic
# 6. fast paths    SUMO reuse (traci.load) + state cache leave learning unchanged
# 7. old layout    run_client.flowerClient (one vehicle per SUMO) runs the same loop

import os
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

import shared_train
from dqnagent import dqnAgent
from run_client import generate_lanedetectionfile, get_alldets
from shared_env import SharedSumoEnv
from utils import calculate_connections, checkSumo, get_alledges

NET, DET, SUMOCFG, DEST = 'Net/dqn.net.xml', 'Add/dqn.det.xml', 'dqn.sumocfg', 'E15'
ROUTE = 'Rou/eval_shared_p3_scenario_0.rou.xml'
START_EDGES = [['E0'], ['E1'], ['E3'], ['E11'], ['E8']]
T_MAX = 600.0

failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def setup():
    _, sumolib = checkSumo()
    edgelists = get_alledges(NET)
    conn = calculate_connections(edgelists, NET)
    generate_lanedetectionfile(NET, DET)
    if not os.path.exists(ROUTE):
        shared_train.generate_eval_scenario(NET, ROUTE, seed=1000)
    cfg = {'use_gui': False, 'start': True, 'quit_on_end': True,
           'start_edges': START_EDGES, 'veh_id': 0}
    return sumolib.checkBinary('sumo'), edgelists, get_alldets(edgelists), conn, cfg


def make_slots(edgelists, conn, n, seed0, epsilon, small_batches=False):
    slots = []
    for i in range(n):
        agent = dqnAgent(f'veh{i}', seed0 + i, edgelists, conn, 94, 3, 3, '/tmp')
        agent.epsilon = epsilon
        if small_batches:
            agent.train_start = agent.batch_size = 8   # gradient steps in a short test
        slots.append(shared_train.AgentSlot(f'veh{i}', agent, START_EDGES[i]))
    return slots


def choose(slot, state, valid):
    return slot.agent.get_action(state, valid)


def test_state_cache(binary, edgelists, alldets, conn, cfg):
    env = SharedSumoEnv(binary, SUMOCFG, edgelists, alldets, conn, DEST, 94, 3, cfg)
    bad = 0
    for episode in range(2):                      # second episode goes through traci.load
        env.start(ROUTE)
        for t in range(300):
            env.step()
            if t % 50 == 0:
                for e in edgelists:
                    bad += not np.array_equal(env.get_state_np(e)[0],
                                              np.asarray(env.get_state(None, e), dtype=np.float32))
    env.close()
    check('cached state == dqnEnv.get_state (fresh start and reload)', bad == 0, f'{bad} differ')

    env = SharedSumoEnv(binary, SUMOCFG, edgelists, alldets, conn, DEST, 94, 3,
                        dict(cfg, state_encoding='onehot'))
    env.start(ROUTE)
    for _ in range(300):
        env.step()
    n, ok = len(edgelists), True
    for k, e in enumerate(edgelists):
        x = env.get_state_np(e)[0]
        ok &= (x.shape[0] == SharedSumoEnv.state_dim('onehot', n) and x[k] == 1.0 and x[:n].sum() == 1.0
               and x[n:].min() >= 0.0 and x[2 * n:].max() <= 1.5)
    env.close()
    check('onehot state: current edge one-hot, occupancy and relative speed in range', ok,
          f'{SharedSumoEnv.state_dim("onehot", n)} inputs')


def test_multi_agent(binary, edgelists, alldets, conn, cfg):
    slots = make_slots(edgelists, conn, 4, 100, 0.5, small_batches=True)
    transitions = {s.veh: [] for s in slots}
    decisions = {s.veh: [] for s in slots}
    episode = 0
    for s in slots:
        orig = s.agent.append_sample

        def rec(state, action, reward, next_state, done, next_valid=None, veh=s.veh, orig=orig):
            transitions[veh].append((np.array(state).ravel(), action, reward,
                                     np.array(next_state).ravel(), done, episode))
            orig(state, action, reward, next_state, done, next_valid=next_valid)
        s.agent.append_sample = rec

    orig_decide = shared_train._decide

    def rec_decide(env, slot, curedge, state, choose_action, learn):
        orig_decide(env, slot, curedge, state, choose_action, learn)
        if slot.pending is not None:
            decisions[slot.veh].append((curedge, slot.pending[2], np.array(state).ravel(), episode))
    shared_train._decide = rec_decide

    env = SharedSumoEnv(binary, SUMOCFG, edgelists, alldets, conn, DEST, 94, 3, cfg)
    shared_steps, total_steps = 0, 0
    orig_env_step = env.step

    def rec_env_step():
        nonlocal shared_steps, total_steps
        orig_env_step()
        roads = [env.road(s.veh) for s in slots if not s.done]
        roads = [r for r in roads if r and env.is_edge(r)]
        total_steps += 1
        shared_steps += len(roads) != len(set(roads))
    env.step = rec_env_step

    ends = []
    try:
        # short limit in the last episode so that time-limit cuts occur too
        for episode, t_max in enumerate([T_MAX, T_MAX, 60.0]):
            shared_train.run_episode(env, slots, ROUTE, choose, True, t_max)
            ends += [(s.veh, episode, s.end_reason) for s in slots]
    finally:
        shared_train._decide = orig_decide
    env.close()

    bad_edges = [(v, c, ne) for v, ds in decisions.items() for c, ne, _, _ in ds if ne not in conn[c]]
    check('chosen edges are real connections', not bad_edges, f'{bad_edges[:3]}')

    counts_ok = all(len(transitions[v]) == len(decisions[v]) == len(s.agent.memory)
                    for v, s in zip(transitions, slots))
    check('one transition per decision, each agent stores only its own',
          counts_ok, str({v: (len(transitions[v]), len(decisions[v])) for v in transitions}))

    mismatches, compared, bad_terminal = 0, 0, 0
    reasons = {(v, ep): r for v, ep, r in ends}
    for v in transitions:
        tr, ds = transitions[v], decisions[v]
        for k in range(len(tr)):
            if k + 1 < len(ds) and ds[k + 1][3] == tr[k][5]:
                compared += 1
                mismatches += tr[k][4] or not np.allclose(tr[k][3], ds[k + 1][2])
            else:                                 # last transition of the episode
                expect_terminal = reasons[(v, tr[k][5])] != 'time_limit'
                bad_terminal += bool(tr[k][4]) != expect_terminal
    check('next_state == state at the next decision', compared > 0 and mismatches == 0,
          f'{compared} compared, {mismatches} mismatched')
    check('final transition: terminal on arrival, bootstrap on time limit',
          bad_terminal == 0 and any(r == 'time_limit' for _, _, r in ends),
          f'{bad_terminal} wrong; end reasons {sorted(set(r for _, _, r in ends))}')

    steps = {s.veh: s.agent.global_steps for s in slots}
    check('gradient steps ran for every agent', all(v > 0 for v in steps.values()), str(steps))
    check('agents share roads', shared_steps > 0,
          f'{shared_steps}/{total_steps} steps with 2+ agents on the same edge')


def test_reward_time_metrics(binary, edgelists, alldets, conn, cfg):
    C = shared_train.COL
    t_max = 200.0                                 # low, so arrivals and time-limit cuts both occur
    slots = make_slots(edgelists, conn, 4, 300, 0.2)
    env = SharedSumoEnv(binary, SUMOCFG, edgelists, alldets, conn, DEST, 94, 3, cfg)
    free_flow = env.free_flow_times(START_EDGES[:4])
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    rows, sums, scheduled, roads_ok = [], [], [], True
    for ep in range(6):
        dep = shared_train.draw_departures(rng, [s.veh for s in slots], 100, 60)
        shared_train.run_episode(env, slots, ROUTE, choose, False, t_max,
                                 os.path.join(tmp, f'ti{ep}.xml'), dep)
        for s in slots:
            row = shared_train.episode_row(s, 1, ep, free_flow)
            rows.append(row)
            scheduled.append(dep[s.veh])
            sums.append((s.reward_sum - s.penalty_sum, row[C['Travel_Time']]))
            if s.road_rows:
                roads_ok &= (abs(sum(r[3] for r in s.road_rows) - row[C['Travel_Time']]) < 1e-6
                             and abs(sum(r[4] for r in s.road_rows) - row[C['Waiting_Time']]) < 1e-6)
    env.close()                                   # tripinfo files are complete now

    sumo_cmp = []
    for ep in range(6):
        trips = {t.get('id'): t for t in ET.parse(os.path.join(tmp, f'ti{ep}.xml')).getroot().iter('tripinfo')}
        for r in rows[ep * 4:(ep + 1) * 4]:
            if r[C['Arrived']] == 1 and r[C['Vehicle']] in trips:   # SUMO's own waiting time
                sumo_cmp.append((r[C['Waiting_Time']], float(trips[r[C['Vehicle']]].get('waitingTime'))))

    by_reason = {}
    for r in rows:
        by_reason[r[C['End_Reason']]] = by_reason.get(r[C['End_Reason']], 0) + 1
    check('departures: no agent enters before its scheduled time (warm-up 100 s, window 60 s)',
          all(r[C['Depart_Time']] >= sch and r[C['Depart_Delay']] >= 0 and 100 <= sch <= 160
              for r, sch in zip(rows, scheduled)),
          f"depart delay max {max(r[C['Depart_Delay']] for r in rows):.0f}s")
    check('rewards without penalties sum to -Travel_Time',
          all(abs(a + b) < 1e-6 for a, b in sums), f'{len(sums)} vehicle-episodes')
    check(f'time limit ({t_max:.0f}s) counted from the scheduled departure',
          all(r[C['Depart_Time']] + r[C['Travel_Time']] <= sch + t_max for r, sch in zip(rows, scheduled))
          and all(r[C['Depart_Time']] + r[C['Travel_Time']] == sch + t_max
                  for r, sch in zip(rows, scheduled) if r[C['End_Reason']] == 'time_limit'),
          f'end reasons {by_reason}')
    arrived = [r for r in rows if r[C['Arrived']] == 1]
    check('free-flow time <= travel time of every arrived agent',
          arrived and all(r[C['Free_Flow_Time']] <= r[C['Travel_Time']] + 1e-6 for r in arrived),
          f'free-flow {free_flow}')
    check('metrics consistent (waiting <= time loss <= travel, speed <= 15 m/s, length > 0)',
          all(r[C['Waiting_Time']] <= r[C['Time_Loss']] + 1.0 and r[C['Time_Loss']] <= r[C['Travel_Time']] + 1.0
              and 0 < r[C['Route_Length']] and r[C['Avg_Speed']] <= 15.0 + 1e-6 for r in arrived),
          f'{len(arrived)} arrived rows')
    check('per-road durations and waiting add up to the trip totals', roads_ok)
    check('waiting time matches SUMO tripinfo (arrived agents)',
          bool(sumo_cmp) and all(ours <= theirs + 1e-6 and theirs - ours <= 2.0
                                 for ours, theirs in sumo_cmp),
          f'(ours, SUMO) {sumo_cmp[:4]}')


def test_baselines(binary, edgelists, alldets, conn, cfg):
    """Shortest-route baseline in an empty network reproduces the measured free-flow time;
    both baselines choose real connections and arrive with traffic."""
    C = shared_train.COL
    env = SharedSumoEnv(binary, SUMOCFG, edgelists, alldets, conn, DEST, 94, 3, cfg)
    free_flow = env.free_flow_times(START_EDGES)
    ok, details = True, []
    for k, start in enumerate(START_EDGES):
        slot = shared_train.AgentSlot(f'veh{k}', None, start)
        row = shared_train.evaluate(env, [slot], shared_train.router_policy(env, 'shortest'),
                                    [env.EMPTY_ROUTES], T_MAX, free_flow, 0)[0]
        ok &= row[C['Travel_Time']] == free_flow[start[0]]
        details.append(f"{start[0]} {row[C['Travel_Time']]:.0f}/{free_flow[start[0]]:.0f}")
    check('shortest baseline, empty network: travel time == free-flow time', ok, ', '.join(details))

    slots = [shared_train.AgentSlot(f'veh{i}', None, START_EDGES[i % 5]) for i in range(10)]
    dep = shared_train.draw_departures(np.random.default_rng(1), [s.veh for s in slots], 100, 100)
    for mode in ('shortest', 'reroute'):
        rows = shared_train.evaluate(env, slots, shared_train.router_policy(env, mode), [ROUTE],
                                     T_MAX, free_flow, 0, departures=lambda j: dep)
        check(f'{mode} baseline with traffic: all 10 arrive, no revisits',
              all(r[C['Arrived']] == 1 and r[C['Revisits']] == 0 for r in rows),
              f"travel mean {np.mean([r[C['Travel_Time']] for r in rows]):.1f}s")
    env.close()


def test_fast_paths(binary, edgelists, alldets, conn, cfg, episodes=25):
    """Reusing SUMO (traci.load) and the per-step state cache must not change learning:
    same agents, same seeds -> same results as fresh SUMO starts with uncached states."""
    def run(fast):
        tf.keras.utils.set_random_seed(7)
        env = SharedSumoEnv(binary, SUMOCFG, edgelists, alldets, conn, DEST, 94, 3, cfg)
        if not fast:
            env.get_state_np = lambda e: np.asarray(env.get_state(None, e), dtype=np.float32)[None, :]
        slots = make_slots(edgelists, conn, 4, 50, 0.3)
        out = []
        for ep in range(episodes):
            if not fast:
                env.close()
            shared_train.run_episode(env, slots, ROUTE, choose, True, T_MAX)
            out.append([(s.reward_sum, s.end_time, s.waiting) for s in slots])
        env.close()
        return out
    slow, fast = run(False), run(True)
    first = next((k for k, (a, b) in enumerate(zip(slow, fast)) if a != b), None)
    check(f'SUMO reuse + state cache give identical learning ({episodes} episodes, 4 agents)',
          first is None, f'first different episode: {first}')


def test_old_layout(binary, edgelists, alldets, conn, cfg):
    """run_client.flowerClient (one vehicle per SUMO) trains and evaluates through shared_train."""
    from run_client import flowerClient
    tmp = tempfile.mkdtemp()
    dqn_cfg = dict(cfg, method='none', use_ns3=False, batches_per_agg=1, num_clients=2,
                   out_dir=tmp, max_trip_time=T_MAX, true_rounds=1, background_period=3,
                   warmup=60, depart_window=30)
    agent = dqnAgent('veh1', 1, edgelists, conn, 94, 3, 2, tmp)
    client = flowerClient(agent, False, binary, False, 2, NET, os.path.join(tmp, 't.trip.xml'),
                          os.path.join(tmp, 'r.rou.xml'), 'Add/dqn.add.xml', tmp, tmp, SUMOCFG,
                          None, edgelists, alldets, conn, 'veh1', DEST, 94, 3, dqn_cfg, 1, None)
    weights, _, _ = client.fit(agent.model.get_weights(), {'round': 1})
    rows = open(os.path.join(tmp, 'veh1_train.csv')).read().strip().splitlines()
    for j in range(2):
        path = f'Rou/eval_veh1_scenario_{j}.rou.xml'
        if not os.path.exists(path):
            shared_train.generate_eval_scenario(NET, path, seed=1100 + j)
    loss, n_eval, m = client.evaluate(weights, {'round': 1, 'eval_episodes': 2})
    client._env.close()
    check('old layout: fit() writes one row per episode, evaluate() returns travel-time sums',
          len(rows) == 3 and rows[0].startswith('FL_Round,Episode,Vehicle') and n_eval == 2
          and m['n'] == 2 and abs(loss - m['tt_sum'] / 2) < 1e-9,
          f'{len(rows) - 1} train rows, eval mean travel {loss:.1f}s')


if __name__ == '__main__':
    binary, edgelists, alldets, conn, cfg = setup()
    test_state_cache(binary, edgelists, alldets, conn, cfg)
    test_multi_agent(binary, edgelists, alldets, conn, cfg)
    test_reward_time_metrics(binary, edgelists, alldets, conn, cfg)
    test_baselines(binary, edgelists, alldets, conn, cfg)
    test_fast_paths(binary, edgelists, alldets, conn, cfg)
    test_old_layout(binary, edgelists, alldets, conn, cfg)
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
