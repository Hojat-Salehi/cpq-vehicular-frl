# shared_train.py
# Episode loop for agent vehicles in one SUMO world. Used by both layouts:
# run_shared.py (all vehicles in one world) and run_client.py (one vehicle per
# process, each with its own SUMO).
#
# Departures: background traffic runs alone during a warm-up; each agent then
# departs at its own scheduled time (drawn in [warmup, warmup + depart_window]
# by the caller: re-drawn every training episode, fixed per evaluation scenario).
#
# Decisions: a decision point is a new (non-internal) edge seen before a
# simulation step; the agent decides after that step (timing of the original
# dqnEnv.step). The agent picks the next edge from the current edge's connections.
#
# Reward: minus the measured simulation seconds between two decision points
# (edge entries), plus revisit penalties (-10 on the 2nd visit of an edge,
# -50 from the 3rd). Without penalties the episode's rewards sum to -Travel_Time.
#
# Episode end per agent: it enters the destination edge (terminal), reaches a
# dead end, or leaves the network; or its trip time limit t_max, counted from
# its scheduled departure, runs out ('time_limit', stored as a truncated,
# non-terminal transition so the DQN bootstraps from it; 'not_departed' if it
# never got into the network). Done agents are removed (or arrive); the episode
# ends when all agents are done, and the simulation is reloaded for the next one.

import heapq
import os
import subprocess
import sys

import numpy as np

ROUTE_REGEN_FREQUENCY = 10
STOPPED_SPEED = 0.1    # m/s; SUMO counts a vehicle as waiting below this speed

# One row per vehicle per episode (training: vehK_train.csv, evaluation: vehK_eval.csv)
CSV_HEADER = ['FL_Round', 'Episode', 'Vehicle', 'Start_Edge', 'Arrived', 'End_Reason',
              'Depart_Time', 'Depart_Delay', 'Travel_Time', 'Free_Flow_Time', 'Route_Length',
              'Avg_Speed', 'Waiting_Time', 'Time_Loss', 'Num_Edges', 'Revisits', 'Reward_Sum',
              'Epsilon']
COL = {name: i for i, name in enumerate(CSV_HEADER)}
# One row per vehicle per edge driven (extra log)
ROAD_HEADER = ['Phase', 'FL_Round', 'Episode', 'Vehicle', 'Edge', 'Enter_Time', 'Exit_Time',
               'Duration', 'Waiting_Time']


class AgentSlot:
    """One agent vehicle: its DQN agent plus per-episode bookkeeping."""

    def __init__(self, veh, agent, start_edges, csv_path=None):
        self.veh = veh
        self.agent = agent
        self.start_edges = list(start_edges)
        self.csv_path = csv_path
        self.reset()

    def reset(self, scheduled=0.0):
        self.scheduled = scheduled  # requested departure time
        self.started = False
        self.done = False
        self.end_reason = None
        self.depart_time = None     # actual entry into the network
        self.end_time = None
        self.t_last = None          # start of the open reward interval (last edge entry)
        self.pending = None         # (state, action, nextedge) of the open transition
        self.beforeedge = None
        self.visited = {}
        self.reward_sum = 0.0
        self.penalty_sum = 0.0
        self.revisits = 0
        self.num_edges = 0
        self.waiting = 0.0
        self.distance = float('nan')
        self.time_loss = float('nan')
        self.road = None            # [edge, enter_time, waiting] of the edge being driven
        self.road_rows = []

    @property
    def arrived(self):
        return self.end_reason == 'arrived'


def generate_background(net, add, trip, route_file, seed, period=3.0):
    """Background traffic for one episode block (same randomTrips call as dqn_run, but seeded):
    one new vehicle every `period` seconds for an hour."""
    tool = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    cmd = [
        sys.executable, tool, '-n', net, '-o', trip, '-r', route_file,
        '-b', '0', '-e', '3600', '--period', str(period), '--additional-file', add,
        '--trip-attributes',
        "departLane='best' type='type1' departSpeed='max' departPos='random'",
        '--seed', str(int(seed)),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_eval_scenario(net, route_file, seed, period=3.0):
    """Static evaluation traffic (randomTrips). Written to a temporary
    file and renamed, so runs started in parallel never read a half-written scenario."""
    tool = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
    tmp = f'{route_file}.{os.getpid()}.tmp.rou.xml'
    cmd = [sys.executable, tool, '-n', net, '-r', tmp,
           '-o', route_file.replace('.rou.xml', f'.{os.getpid()}.trip.xml'),
           '-e', '3600', '--period', str(period), '--seed', str(int(seed))]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, route_file)
    for leftover in (tmp.replace('.rou.xml', '.rou.alt.xml'),
                     route_file.replace('.rou.xml', f'.{os.getpid()}.trip.xml')):
        if os.path.exists(leftover):
            os.remove(leftover)


def draw_departures(rng, vehs, warmup, window):
    """{veh: departure time}, uniform in [warmup, warmup + window], whole seconds."""
    return {v: float(warmup + (rng.integers(0, int(window) + 1) if window > 0 else 0)) for v in vehs}


# ── per-vehicle events ─────────────────────────────────────────────────────────
def _enter_road(slot, edge, t):
    if slot.road is not None:
        e, t0, w = slot.road
        slot.road_rows.append((e, t0, t, t - t0, w))
    slot.road = [edge, t, 0.0] if edge is not None else None


def _close_transition(slot, t, next_state, terminal, learn, next_valid=None):
    """Close the open transition at time t (an edge entry or the episode end)."""
    state, action, nextedge = slot.pending
    slot.pending = None
    reward = -(t - slot.t_last)
    slot.t_last = t

    n = slot.visited.get(nextedge, 0) + 1
    slot.visited[nextedge] = n
    penalty = -10.0 if n == 2 else (-50.0 if n >= 3 else 0.0)
    slot.revisits += n >= 2
    reward += penalty
    slot.penalty_sum += penalty
    slot.reward_sum += reward

    if learn:
        slot.agent.append_sample(state, action, reward, next_state, terminal, next_valid=next_valid)
        if len(slot.agent.memory) >= slot.agent.train_start:
            slot.agent.train_model()


def _end(env, slot, t, reason, learn):
    """Finish a vehicle's episode at time t."""
    if slot.pending is not None:
        terminal = reason != 'time_limit'      # time limit = truncation: bootstrap
        road = env.road(slot.veh)
        edge = road if (reason == 'time_limit' and env.is_edge(road)) else slot.pending[2]
        _close_transition(slot, t, env.get_state_np(edge), terminal, learn,
                          next_valid=env.valid_actions(edge) if env.is_edge(edge) else None)
    if slot.started:
        slot.distance, slot.time_loss = env.trip_stats(slot.veh)
    slot.done = True
    slot.end_reason = reason
    slot.end_time = t
    _enter_road(slot, None, t)
    if reason != 'arrived':
        env.remove_vehicle(slot.veh)


def _decide(env, slot, curedge, state, choose_action, learn):
    valid = env.valid_actions(curedge)
    if not valid:
        _end(env, slot, slot.t_last, 'dead_end', learn)
        return
    slot.curedge = curedge
    action = choose_action(slot, state, valid)
    nextedge = env.dict_connection[curedge][action]
    env.change_target(slot.veh, nextedge)
    slot.pending = (state, action, nextedge)
    slot.beforeedge = curedge
    slot.num_edges += 1


def run_episode(env, slots, route_file, choose_action, learn, t_max, tripinfo=None,
                departures=None, horizon=None, on_step=None):
    """Run one episode with the given slots in one SUMO world.

    departures: {veh: scheduled departure time} (default: all at t=0).
    t_max: trip time limit per agent, counted from its scheduled departure.
    horizon: if given, keep simulating until this time after the agents are done
             (fixed window for background-traffic metrics).
    on_step(t): called after every simulation step with the new time (communication).
    """
    env.start(route_file, tripinfo)
    dt = env.delta_t()
    for s in slots:
        s.reset((departures or {}).get(s.veh, 0.0))
        env.add_vehicle(s.veh, s.start_edges, s.scheduled)

    live = list(slots)
    while live:
        now = env.time()
        # Observe before the step (decision timing of the original dqnEnv.step/reset)
        ready = []
        for s in live:
            if now >= s.scheduled + t_max:
                _end(env, s, now, 'time_limit' if s.started else 'not_departed', learn)
                continue
            road = env.road(s.veh)
            if not s.started:
                if road:                         # inserted into the network
                    ready.append((s, road, now))
                continue
            if road is None:
                _end(env, s, now, 'left', learn)
            elif road == env.destination:
                _end(env, s, now, 'arrived', learn)
            elif env.is_edge(road) and road != s.beforeedge:
                ready.append((s, road, now))

        live = [s for s in live if not s.done]
        if not live:
            break
        env.step()
        if on_step is not None:
            on_step(env.time())

        for s in live:                           # stopped time, per vehicle and per edge
            if s.started and env.speed(s.veh) < STOPPED_SPEED:
                s.waiting += dt
                if s.road is not None:
                    s.road[2] += dt

        for s, road, t_seen in ready:
            now_road = env.road(s.veh)
            curedge = now_road if env.is_edge(now_road) else road   # avoid internal junction lanes
            state = env.get_state_np(curedge)
            if not s.started:
                s.started = True
                s.depart_time = s.t_last = t_seen
                s.visited = {curedge: 1}
            else:
                _close_transition(s, t_seen, state, False, learn, next_valid=env.valid_actions(curedge))
            _enter_road(s, curedge, t_seen)
            _decide(env, s, curedge, state, choose_action, learn)

        live = [s for s in live if not s.done]

    if horizon is not None:
        while env.time() < horizon:
            env.step()


def episode_row(slot, round_num, episode, free_flow):
    started = slot.depart_time is not None
    travel = (slot.end_time - slot.depart_time) if started else float('nan')
    eps = slot.agent.epsilon if slot.agent is not None else 0.0
    return [round_num, episode, slot.veh, slot.start_edges[0], int(slot.arrived), slot.end_reason,
            slot.depart_time if started else float('nan'),
            (slot.depart_time - slot.scheduled) if started else float('nan'),
            travel, free_flow.get(slot.start_edges[0], float('nan')),
            round(slot.distance, 1), round(slot.distance / travel, 3) if travel else float('nan'),
            slot.waiting, round(slot.time_loss, 2), slot.num_edges, slot.revisits,
            round(slot.reward_sum, 2), round(eps, 6)]


def road_rows(slot, phase, round_num, episode):
    return [[phase, round_num, episode, slot.veh, e, t0, t1, d, w] for e, t0, t1, d, w in slot.road_rows]


# ── policies ───────────────────────────────────────────────────────────────────
def epsilon_greedy_policy(slot, state, valid):
    return slot.agent.get_action(state, valid)


def own_greedy_policy(slot, state, valid):
    """Each vehicle's own DQN, no exploration (communication drives)."""
    q = slot.agent.model(state, training=False).numpy()[0]
    return valid[int(np.argmax(q[valid]))]


def greedy_policy(model):
    """The (global) DQN model, no exploration."""
    def choose(slot, state, valid):
        q = model(state, training=False).numpy()[0]
        masked = np.full(q.shape[0], -np.inf)
        masked[valid] = q[valid]
        return int(np.argmax(masked))
    return choose


def router_policy(env, mode):
    """Non-learning baselines, deciding at the same points as the agents.

    'shortest': next edge of the fastest route at free-flow speed (static).
    'reroute':  next edge of the fastest route under the current travel time of
                every edge (SUMO's estimate), re-planned at every edge.
    The cost of a route is the sum of the travel times of the edges entered
    after the current one, up to entering the destination (as Travel_Time).
    """
    conn = {e: [n for n in nxt if n] for e, nxt in env.dict_connection.items()}
    preds = {}
    for e, nxt in conn.items():
        for n in nxt:
            preds.setdefault(n, []).append(e)

    def cost_to_go(weight):
        dist = {env.destination: 0.0}
        heap = [(0.0, env.destination)]
        while heap:
            d, e = heapq.heappop(heap)
            if d > dist.get(e, np.inf):
                continue
            w = 0.0 if e == env.destination else weight(e)
            for p in preds.get(e, []):
                if d + w < dist.get(p, np.inf):
                    dist[p] = d + w
                    heapq.heappush(heap, (d + w, p))
        return dist

    cache = {}

    def choose(slot, state, valid):
        if mode == 'shortest':
            if 'static' not in cache:
                cache['static'] = cost_to_go(env.free_flow_edge_time)
            dist, weight = cache['static'], env.free_flow_edge_time
        else:
            t = env.time()
            if cache.get('t') != t:
                times = {e: env.current_edge_time(e) for e in env.edgelists}
                cache.update(t=t, dist=cost_to_go(times.get), times=times)
            dist, weight = cache['dist'], cache['times'].get
        options = env.dict_connection[slot.curedge]
        costs = [(0.0 if options[a] == env.destination else weight(options[a]))
                 + dist.get(options[a], np.inf) for a in valid]
        return valid[int(np.argmin(costs))]
    return choose


# ── training and evaluation ────────────────────────────────────────────────────
def train_round(env, slots, num_episode, round_num, background, write_row, t_max, free_flow,
                road_log=None, departures=None):
    """Local training for one FL round: num_episode episodes with all slots together.

    background(episode) -> route file; departures(episode) -> {veh: time} or None.
    Returns per-episode lists of CSV rows.
    """
    history = []
    for episode in range(num_episode):
        run_episode(env, slots, background(episode), epsilon_greedy_policy, True, t_max,
                    departures=departures(episode) if departures else None)
        rows = []
        for s in slots:
            s.agent.decay_epsilon()
            row = episode_row(s, round_num, episode, free_flow)
            rows.append(row)
            write_row(s.csv_path, row)
            if road_log:
                for r in road_rows(s, 'train', round_num, episode):
                    write_row(road_log, r)
        history.append(rows)
        travel = [r[COL['Travel_Time']] for r in rows]
        print(f"[EPISODE] round {round_num} episode {episode}: "
              f"travel mean={np.nanmean(travel):.0f}s median={np.nanmedian(travel):.0f}s "
              f"arrived={sum(r[COL['Arrived']] for r in rows)}/{len(rows)}", flush=True)
    return history


def evaluate(env, slots, choose, scenario_files, t_max, free_flow, round_num,
             write_row=None, eval_csv=None, road_log=None, tripinfo=None,
             departures=None, horizon=None):
    """Evaluate one policy (greedy_policy(model) or router_policy) controlling every slot.

    eval_csv(slot) -> path of that vehicle's evaluation CSV; tripinfo(scenario) ->
    path for SUMO's trip summary; departures(scenario) -> {veh: time}; horizon:
    simulate every scenario to this time (fixed window for background metrics).
    Returns all rows.
    """
    rows = []
    for j, route_file in enumerate(scenario_files):
        run_episode(env, slots, route_file, choose, False, t_max,
                    tripinfo(j) if tripinfo else None,
                    departures(j) if departures else None, horizon)
        for s in slots:
            row = episode_row(s, round_num, j, free_flow)
            row[COL['Epsilon']] = 0.0
            rows.append(row)
            if write_row and eval_csv:
                write_row(eval_csv(s), row)
            if write_row and road_log:
                for r in road_rows(s, 'eval', round_num, j):
                    write_row(road_log, r)
    return rows


def summarize(rows):
    """Exact sums so that several processes' evaluations can be combined."""
    tt = np.array([r[COL['Travel_Time']] for r in rows], dtype=float)
    return {'n': float(len(rows)), 'tt_sum': float(np.nansum(tt)), 'tt_sumsq': float(np.nansum(tt ** 2)),
            'arrived_sum': float(sum(r[COL['Arrived']] for r in rows)),
            'time_loss_sum': float(np.nansum([r[COL['Time_Loss']] for r in rows])),
            'waiting_sum': float(np.nansum([r[COL['Waiting_Time']] for r in rows]))}


EVAL_SUMMARY_HEADER = ['round', 'travel_time_mean', 'travel_time_std', 'arrival_rate',
                       'time_loss_mean', 'waiting_time_mean', 'n']


def summary_row(round_num, sums):
    """Combine summarize() dicts into one global_eval.csv row."""
    n = sum(s['n'] for s in sums)
    if n == 0:
        return [round_num] + [float('nan')] * 5 + [0]
    mean = sum(s['tt_sum'] for s in sums) / n
    var = max(sum(s['tt_sumsq'] for s in sums) / n - mean ** 2, 0.0)
    return [round_num, round(mean, 2), round(var ** 0.5, 2),
            round(sum(s['arrived_sum'] for s in sums) / n, 4),
            round(sum(s['time_loss_sum'] for s in sums) / n, 2),
            round(sum(s['waiting_sum'] for s in sums) / n, 2), int(n)]


BACKGROUND_HEADER = ['FL_Round', 'Scenario', 'Vehicles', 'Finished', 'Duration_Mean',
                     'Time_Loss_Mean', 'Waiting_Time_Mean', 'Avg_Speed']


def background_csv(tripinfo_dir, out_csv, agent_ids, window_start=0.0):
    """Summarize SUMO tripinfo files (one per evaluation scenario) for the non-agent traffic.

    Counted: every non-agent vehicle that entered the network at or after
    window_start (the end of the warm-up); vehicles still driving at the end of
    the fixed window are included with their values so far. Scenarios are fixed,
    so differences between rounds come only from the agents. FL_Round 0 is the
    same scenarios without agents; other labels (e.g. 'shortest') are baselines.
    The XML files are deleted once summarized.
    """
    import re
    import xml.etree.ElementTree as ET
    rows = []
    for name in os.listdir(tripinfo_dir):
        m = re.match(r'eval_r(\w+?)_s(\d+)\.xml$', name)
        if not m:
            continue
        path = os.path.join(tripinfo_dir, name)
        trips = [t for t in ET.parse(path).getroot().iter('tripinfo')
                 if t.get('id') not in agent_ids and float(t.get('depart')) >= window_start]
        f = lambda k: np.array([float(t.get(k)) for t in trips]) if trips else np.array([np.nan])
        dur = f('duration')
        label = int(m.group(1)) if m.group(1).isdigit() else m.group(1)
        rows.append([label, int(m.group(2)), len(trips),
                     sum(float(t.get('arrival')) >= 0 for t in trips),
                     round(float(np.mean(dur)), 2), round(float(np.mean(f('timeLoss'))), 2),
                     round(float(np.mean(f('waitingTime'))), 2),
                     round(float(np.sum(f('routeLength')) / np.sum(dur)), 3)])
        os.remove(path)
    rows.sort(key=lambda r: (isinstance(r[0], int), r[0] if isinstance(r[0], int) else 0, str(r[0]), r[1]))
    with open(out_csv, 'w') as fh:
        fh.write(','.join(BACKGROUND_HEADER) + '\n')
        for r in rows:
            fh.write(','.join(str(x) for x in r) + '\n')
    try:
        os.rmdir(tripinfo_dir)
    except OSError:
        pass
