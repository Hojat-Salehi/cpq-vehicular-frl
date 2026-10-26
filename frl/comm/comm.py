# Model exchange over ns-3 (ns3.use_ns3: true). Each round:
#   1. downlink drive: the RSU sends the signed global model to the vehicles in range
#   2. local training
#   3. uplink drive: vehicles that trained from the current model send their signed, privatized update
# A vehicle is in range when it hears enough RSU beacons (ns3.beacon_*). The RSU serves one transfer at
# a time (ns3.rsu_max_concurrent); failed transfers are retried while the vehicle is in range. Received
# packets are verified (signature, round, vehicle) before use. With privacy.cpq_group_key: ml_kem the
# CPQ group setup (frl/privacy/group_setup.py) also runs over ns-3 before round 1.
# Output: comm.csv, one row per transfer attempt.

import csv
import math
import os
from collections import namedtuple

from frl.comm.ns3_link import Ns3Link
from frl.comm.pqc_crypto import PQCCrypto, verify_and_unpack_packet
from frl.env.shared_train import own_greedy_policy, run_episode

COMM_HEADER = ['FL_Round', 'Direction', 'Vehicle', 'Msg_Id', 'Attempt', 'Start_Time', 'End_Time',
               'Duration', 'Outcome', 'Wire_Bytes', 'Dist_Start', 'Dist_End', 'Queue_Wait', 'Message']

# One transfer: 'uplink' (vehicle -> RSU) or 'downlink' (RSU -> vehicle) of payload; key
# identifies it for attempt counting; label/kind go to comm.csv (Direction / Message).
Item = namedtuple('Item', 'direction payload key label kind')


class _ModelWork:
    """Model drive: one signed packet per vehicle in one direction."""

    def __init__(self, manager, direction, round_num, packets):
        self.m, self.direction, self.round, self.pending = manager, direction, round_num, dict(packets)
        self.delivered = {}

    def next_item(self, veh):
        if veh not in self.pending:
            return None
        return Item(self.direction, self.pending[veh], veh, self.direction, 'model')

    def on_result(self, veh, item, ok, payload, t):
        if not ok:
            return 'failed'
        weights = self.m._verify(payload, veh, self.round, self.direction)
        if weights is None:
            return 'bad_signature'
        self.delivered[veh] = weights
        self.pending.pop(veh, None)
        return 'delivered'


class _SetupWork:
    """CPQ group setup relayed by the RSU (state machine of group_setup.py messages)."""

    def __init__(self, vehicles, group_secret):
        self.v = vehicles                               # veh -> group_setup.Vehicle
        self.roster = list(vehicles)
        self.D = self.roster[0]
        self.others = self.roster[1:]
        self.key_up = dict(self.v[self.D].key_messages(group_secret))   # u -> KEY, still at D
        self.rsu_key = {}                               # u -> KEY stored at the RSU
        self.confirm_up = {}                            # u -> CONFIRM, still at u
        self.rsu_confirm = {}                           # u -> CONFIRM stored at the RSU
        self.d_confirms = {}                            # u -> CONFIRM received by D
        self.ok_up = None                               # SETUP_OK, still at D
        self.rsu_ok = None                              # SETUP_OK stored at the RSU
        self.done = set()                               # vehicles that verified SETUP_OK

    def complete(self):
        return self.rsu_ok is not None and self.done >= set(self.others)

    def next_item(self, veh):
        if veh == self.D:
            for u in self.others:
                if u in self.key_up:
                    return Item('uplink', self.key_up[u], ('KEY:up', u), 'setup', 'KEY:up')
            for u in self.others:
                if u in self.rsu_confirm and u not in self.d_confirms:
                    return Item('downlink', self.rsu_confirm[u], ('CONFIRM:down', u), 'setup', 'CONFIRM:down')
            if self.ok_up is not None and self.rsu_ok is None:
                return Item('uplink', self.ok_up, ('SETUP_OK:up',), 'setup', 'SETUP_OK:up')
            return None
        if veh in self.rsu_key and veh not in self.confirm_up and veh not in self.rsu_confirm:
            return Item('downlink', self.rsu_key[veh], ('KEY:down', veh), 'setup', 'KEY:down')
        if veh in self.confirm_up:
            return Item('uplink', self.confirm_up[veh], ('CONFIRM:up', veh), 'setup', 'CONFIRM:up')
        if self.rsu_ok is not None and veh not in self.done:
            return Item('downlink', self.rsu_ok, ('SETUP_OK:down', veh), 'setup', 'SETUP_OK:down')
        return None

    def on_result(self, veh, item, ok, payload, t):
        if not ok:
            return 'failed'
        kind, *who = item.key
        if kind == 'KEY:up':                            # RSU now holds the KEY for who[0]
            self.rsu_key[who[0]] = payload
            del self.key_up[who[0]]
        elif kind == 'KEY:down':                        # vehicle verifies, derives keys, confirms
            self.confirm_up[veh] = self.v[veh].receive_key(payload)
            del self.rsu_key[veh]
        elif kind == 'CONFIRM:up':
            self.rsu_confirm[veh] = payload
            del self.confirm_up[veh]
        elif kind == 'CONFIRM:down':                    # designated vehicle collects confirmations
            self.d_confirms[who[0]] = payload
            if len(self.d_confirms) == len(self.others):
                self.ok_up = self.v[self.D].check_confirmations(self.d_confirms)
        elif kind == 'SETUP_OK:up':
            self.rsu_ok = payload
        elif kind == 'SETUP_OK:down':
            self.v[veh].receive_setup_ok(payload)
            self.done.add(veh)
        return 'delivered'


class CommManager:
    def __init__(self, cfg, vehs, out_dir, rsu_pos):
        ns3 = cfg['ns3']
        self.vehs = list(vehs)
        self.index = {v: int(v[3:]) for v in self.vehs}
        self.rsu = rsu_pos
        self.radius = float(ns3['coverage_radius'])
        self.max_concurrent = int(ns3.get('rsu_max_concurrent', 1))
        self.discovery = ns3.get('discovery', 'beacon')
        self.beacon_timeout = float(ns3.get('beacon_timeout', 0.5))
        self.beacon_min = int(ns3.get('beacon_min', 3))
        self.setup_max_drives = int(ns3.get('setup_max_drives', 20))
        self.schedule_step = float(ns3.get('schedule_step', 1.0))
        if not 0 < self.schedule_step <= 1.0:
            raise ValueError(f'ns3.schedule_step must be in (0, 1] s, not {self.schedule_step}')
        if self.discovery not in ('beacon', 'distance'):
            raise ValueError(f"ns3.discovery must be 'beacon' or 'distance', not {self.discovery!r}")
        self.crypto = PQCCrypto({'scheme': cfg['pqc']['scheme'], 'key_dir': cfg['pqc']['key_dir']})
        self.key_dir = cfg['pqc']['key_dir']
        beacons = float(ns3.get('beacon_interval', 0.1)) if self.discovery == 'beacon' else 0.0
        self.link = Ns3Link(max(self.index.values()) + 1, rsu_pos, ns3['ns3_dir'],
                            ns3.get('fl_sim_args', '') or '', log_path=os.path.join(out_dir, 'ns3.log'),
                            beacon_interval=beacons)
        self.csv_path = os.path.join(out_dir, 'comm.csv')
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(COMM_HEADER)

    def close(self):
        self.link.close()

    def distance(self, x, y):
        return math.hypot(x - self.rsu[0], y - self.rsu[1])

    def packet(self, veh, round_num, weights, direction):
        """Signed wire packet: the RSU signs downlinks, the vehicle signs its uplink."""
        return self.crypto._build_packet(veh, round_num, weights, direction)['packet']

    def _verify(self, payload, veh, round_num, direction):
        try:
            header, weights = verify_and_unpack_packet(payload, key_dir=self.key_dir)
        except Exception:
            return None
        if (str(header.get('veh')) != veh or int(header.get('round', -1)) != int(round_num)
                or header.get('direction') != direction):
            return None
        return weights

    def run_drive(self, env, slots, route_file, departures, t_max, direction, round_num, packets):
        """One model drive. packets: {veh: wire bytes} still to deliver.

        Returns {veh: weights} for packets delivered and verified before the end of the drive.
        """
        work = _ModelWork(self, direction, round_num, packets)
        self._drive(env, slots, route_file, departures, t_max, round_num, work)
        return work.delivered

    def run_setup(self, env, slots, route_file_fn, departures_fn, t_max, vehicles, group_secret, log=print):
        """CPQ group setup over ns-3 in setup drives (FL_Round 0 in comm.csv) until every vehicle
        verified SETUP_OK, at most ns3.setup_max_drives drives; raises RuntimeError otherwise.
        vehicles: {veh: group_setup.Vehicle}. Returns the number of drives used."""
        work = _SetupWork(vehicles, group_secret)
        for drive in range(1, self.setup_max_drives + 1):
            self._drive(env, slots, route_file_fn(drive), departures_fn(drive), t_max, 0, work)
            log(f'[SETUP] drive {drive}: KEY at RSU {len(work.rsu_key) + len(work.confirm_up) + len(work.rsu_confirm)}'
                f'/{len(work.others)} delivered-or-later, confirmations at the designated vehicle '
                f'{len(work.d_confirms)}/{len(work.others)}, SETUP_OK verified {len(work.done)}/{len(work.others)}')
            if work.complete():
                return drive
        raise RuntimeError(f'CPQ group setup incomplete after {self.setup_max_drives} drives '
                           f'(SETUP_OK verified by {len(work.done)}/{len(work.others)} vehicles); aborting as '
                           f'required by the protocol')

    def _drive(self, env, slots, route_file, departures, t_max, round_num, work):
        """One non-learning drive: move work's items through the RSU while vehicles are in range."""
        link = self.link
        link.reset()
        offset = math.ceil(link.now()) + 1.0         # ns-3 time of SUMO t = 0 in this drive
        in_flight = {}                                # msg_id -> (veh, item, attempt, start_t, dist_start, wait)
        queued = {}                                   # veh -> time it joined the RSU queue
        attempts = {}
        parked = set(self.index[v] for v in self.vehs)
        last_pos = {}
        rows = []

        def finish(msg_id, t, outcome):
            veh, item, attempt, t0, d0, wait = in_flight.pop(msg_id)
            pos = last_pos.get(veh)
            d1 = self.distance(*pos[:2]) if pos else float('nan')
            t = round(t, 3)
            rows.append([round_num, item.label, veh, msg_id, attempt, t0, t, round(t - t0, 3), outcome,
                         len(item.payload), round(d0, 1), round(d1, 1), wait, item.kind])

        def collect(t):
            # ns-3 catches up to drive time t; finished transfers free the RSU
            for ev in link.advance(offset + t):
                if ev.msg_id not in in_flight:
                    continue
                veh, item = in_flight[ev.msg_id][:2]
                outcome = work.on_result(veh, item, ev.ok, ev.payload, ev.t - offset)
                finish(ev.msg_id, ev.t - offset, outcome)
                queued.pop(veh, None)                 # back of the queue (re-joins below)

        def schedule(t, motion):
            # RSU queue: vehicles in coverage with something to transfer, first come first
            # served; leaving coverage drops a vehicle from the queue
            busy = {info[0] for info in in_flight.values()}
            if self.discovery == 'beacon':
                heard = link.heard(self.beacon_timeout, self.beacon_min)
                in_range = {v for v in motion if self.index[v] in heard}
            else:
                in_range = {v for v, (x, y, _, _) in motion.items() if self.distance(x, y) <= self.radius}
            in_cov = {v for v in in_range if v not in busy and work.next_item(v) is not None}
            for v in list(queued):
                if v not in in_cov:
                    del queued[v]
            for v in sorted(in_cov - set(queued)):
                queued[v] = t
            free = self.max_concurrent - len(in_flight)
            for v in sorted(queued, key=lambda v: (queued[v], v))[:max(free, 0)]:
                x, y = motion[v][:2]
                item = work.next_item(v)
                attempts[item.key] = attempts.get(item.key, 0) + 1
                send = link.send_update if item.direction == 'uplink' else link.send_global
                msg_id = send(v, round_num, item.payload)
                in_flight[msg_id] = (v, item, attempts[item.key], t, self.distance(x, y),
                                     round(t - queued.pop(v), 3))

        prev = {}                                     # SUMO time and motion of the previous step

        def on_step(t):
            # 1) RSU decisions between SUMO steps, every ns3.schedule_step s (none with 1.0):
            #    ns-3 moves the vehicles with the velocities set at the previous step; positions for
            #    discovery by distance and for comm.csv are extrapolated the same way
            if prev and self.schedule_step < t - prev['t']:
                t0, m0 = prev['t'], prev['motion']
                n = max(1, round((t - t0) / self.schedule_step))
                for k in range(1, n):
                    s = round(t0 + k * (t - t0) / n, 6)
                    dt = s - t0
                    motion_s = {v: (x + vx * dt, y + vy * dt, vx, vy) for v, (x, y, vx, vy) in m0.items()}
                    last_pos.update(motion_s)
                    collect(s)
                    schedule(s, motion_s)

            # 2) ns-3 catches up to SUMO time t (motion set at the previous step)
            collect(t)

            # 3) positions and velocities at t
            active = [s.veh for s in slots if s.started and not s.done]
            motion = env.motion(active)
            last_pos.update(motion)
            link.set_motion({self.index[v]: m for v, m in motion.items()})
            now_parked = set(self.index[v] for v in self.vehs) - set(self.index[v] for v in motion)
            link.park(sorted(now_parked - parked))
            parked.clear()
            parked.update(now_parked)
            prev.update(t=t, motion=motion)

            # 4) RSU queue at t
            schedule(t, motion)

        run_episode(env, slots, route_file, own_greedy_policy, False, t_max,
                    departures=departures, on_step=on_step)
        t_end = env.time()
        for msg_id in list(in_flight):                # still running at the deadline
            finish(msg_id, t_end, 'deadline')
        link.reset()
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerows(rows)
