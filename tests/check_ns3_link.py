# tests/check_ns3_link.py
# Checks for ns-3 lockstep mode (fl-sim --lockstep, ns3_link.py) and the communication drives
# (comm.py). Needs the fl-sim binary (ns3_integration/README_NS3.md).
#   python tests/check_ns3_link.py        (inside sumo_env, from the repo root)
#
# 1. motion       positions and constant-velocity motion set from Python are what ns-3 uses
# 2. lossless     with propModel=none every transfer arrives intact; delay grows with size at
#                 roughly the 6 Mbps PHY rate
# 3. range        a parked (out-of-range) vehicle never delivers
# 4. distance     default channel: delivery close to the RSU, none far away
# 5. determinism  the same commands give the same events
# 6. signatures   a tampered packet is rejected, a correct one accepted
# 7. beacons      RSU beacons are heard close to the RSU, not far away or when parked; a drive
#                 with beacon discovery starts transfers only for vehicles that heard beacons
# 8. drive        a communication drive in SUMO (distance discovery): transfers start only within the coverage radius
#                 and one at a time (RSU queue),
#                 every delivered packet is verified, comm.csv rows are consistent

import csv
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

from ns3_link import Ns3Link

NS3_DIR = '~/ns-3.46'
RSU = (300.0, 100.0)
failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def collect(link, t_until, step=1.0):
    events, t = [], link.now()
    while t < t_until:
        t = min(t + step, t_until)
        events += link.advance(t)
    return events


def test_motion_lossless_range():
    tmp = tempfile.mkdtemp()
    link = Ns3Link(4, RSU, NS3_DIR, '--propModel=none', log_path=os.path.join(tmp, 'ns3.log'))
    link.set_motion({0: (310, 100, 0, 0), 1: (300, 150, 10, 0), 2: (350, 100, 0, 0)})
    link.advance(1.0)
    x, y, d = link.status(1)
    check('motion: position and constant velocity follow the commands',
          abs(x - 310) < 1e-6 and abs(y - 150) < 1e-6 and abs(link.status(0)[2] - 10) < 1e-6,
          f'veh1 after 1 s at ({x:.1f}, {y:.1f})')

    # one transfer at a time (concurrent transfers share the channel and take longer)
    sizes = {0: 8700, 1: 23000, 2: 112000}
    delay, intact = {}, True
    for i, n in sizes.items():
        payload = os.urandom(n)
        start = link.now()
        send = link.send_global if i == 1 else link.send_update
        m = send(f'veh{i}', 1, payload)
        ev = [e for e in collect(link, start + 3.0) if e.msg_id == m]
        intact &= len(ev) == 1 and ev[0].ok and ev[0].payload == payload
        delay[i] = ev[0].t - start if ev else float('nan')
    airtime = sizes[2] * 8 / 6e6               # payload bits at the 6 Mbps PHY rate
    rate_ok = delay[0] < delay[1] < delay[2] and airtime <= delay[2] <= 2.5 * airtime
    check('lossless: every transfer arrives intact (uplink and downlink)', intact)
    check('lossless: delay grows with size, near the 6 Mbps PHY rate', rate_ok,
          f"8.7 KB {delay[0] * 1000:.0f} ms, 23 KB {delay[1] * 1000:.0f} ms, 112 KB {delay[2] * 1000:.0f} ms "
          f"(112 KB at 6 Mbps = {airtime * 1000:.0f} ms of payload airtime)")

    m = link.send_update('veh3', 1, os.urandom(8700))     # veh3 never placed: parked
    t_send = link.now()
    ev = [e for e in collect(link, t_send + 6.0) if e.msg_id == m]
    check('range: a parked vehicle never delivers', len(ev) == 1 and not ev[0].ok,
          f'failed after {ev[0].t - t_send:.2f} s' if ev else 'no event')
    link.close()


def run_script(fl_sim_args):
    tmp = tempfile.mkdtemp()
    link = Ns3Link(6, RSU, NS3_DIR, fl_sim_args, log_path=os.path.join(tmp, 'ns3.log'))
    dists = [20, 60, 100, 150, 250, 400]
    link.set_motion({i: (RSU[0] + d, RSU[1], 0, 0) for i, d in enumerate(dists)})
    rng = np.random.default_rng(0)
    msgs = {link.send_update(f'veh{i}', 1, rng.bytes(8700)): d for i, d in enumerate(dists)}
    events = collect(link, 20.0)
    link.close()
    return dists, msgs, events


def test_distance_and_determinism():
    dists, msgs, events = run_script('')
    ok = {msgs[e.msg_id]: e.ok for e in events}
    check('distance (default channel): delivered at 20 m, not at 400 m',
          ok.get(20) is True and ok.get(400) is False,
          ', '.join(f'{d} m: {"ok" if ok.get(d) else "lost"}' for d in dists))
    _, _, events2 = run_script('')
    check('determinism: same commands, same events',
          [(e.msg_id, e.ok, e.t) for e in events] == [(e.msg_id, e.ok, e.t) for e in events2])


def test_signatures():
    from pqc.pqc_crypto import PQCCrypto, verify_and_unpack_packet
    crypto = PQCCrypto({'scheme': 'ML-DSA-44', 'key_dir': 'pqc_keys'})
    w = [np.arange(12, dtype=np.float32).reshape(3, 4), np.ones(3, dtype=np.uint8)]
    pkt = crypto._build_packet('veh0', 3, w, 'uplink')['packet']
    hdr, back = verify_and_unpack_packet(pkt, key_dir='pqc_keys')
    good = hdr['round'] == 3 and all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(w, back))
    bad = bytearray(pkt)
    bad[-5] ^= 0xFF
    try:
        verify_and_unpack_packet(bytes(bad), key_dir='pqc_keys')
        rejected = False
    except Exception:
        rejected = True
    check('signatures: intact packet verifies (weights and dtypes), tampered one is rejected',
          good and rejected)


def test_drive():
    """One downlink drive with 4 vehicles in SUMO through CommManager."""
    import shared_train
    from comm import CommManager
    from dqnagent import dqnAgent
    from run_client import get_alldets
    from shared_env import SharedSumoEnv
    from utils import calculate_connections, checkSumo, get_alledges
    _, sumolib = checkSumo()
    net = 'Net/dqn.net.xml'
    edges = get_alledges(net)
    conn = calculate_connections(edges, net)
    cfg = {'use_gui': False, 'start': True, 'quit_on_end': True, 'veh_id': 0,
           'ns3': {'ns3_dir': NS3_DIR, 'fl_sim_args': '--propModel=none', 'coverage_radius': 150,
                   'discovery': 'distance'},
           'pqc': {'scheme': 'ML-DSA-44', 'key_dir': 'pqc_keys'}}
    env = SharedSumoEnv(sumolib.checkBinary('sumo'), 'dqn.sumocfg', edges, get_alldets(edges), conn,
                        'E15', 94, 3, cfg)
    starts = [['E0'], ['E1'], ['E3'], ['E11']]
    slots = [shared_train.AgentSlot(f'veh{i}', dqnAgent(f'veh{i}', i, edges, conn, 94, 3, 1, '/tmp'),
                                    starts[i]) for i in range(4)]
    tmp = tempfile.mkdtemp()
    comm = CommManager(cfg, [s.veh for s in slots], tmp, RSU)
    # 1.6 MB: a transfer lasts a few seconds, so vehicles arriving meanwhile must queue
    w = [np.random.default_rng(1).normal(size=(400, 1000)).astype(np.float32)]
    packets = {s.veh: comm.packet(s.veh, 1, w, 'downlink') for s in slots}
    route = 'Rou/eval_shared_p3_scenario_0.rou.xml'
    # all depart together, so several vehicles are in coverage at the same time
    got = comm.run_drive(env, slots, route, {s.veh: 50.0 for s in slots}, 600, 'downlink', 1, packets)
    comm.close()
    env.close()
    rows = list(csv.DictReader(open(os.path.join(tmp, 'comm.csv'))))
    check('drive: transfers start only within the coverage radius',
          rows and all(float(r['Dist_Start']) <= 150 for r in rows), f'{len(rows)} attempts')
    check('drive (lossless): every vehicle that came within range received a verified model',
          set(got) == {r['Vehicle'] for r in rows}
          and all(np.array_equal(got[v][0], w[0]) for v in got)
          and all(r['Outcome'] == 'delivered' for r in rows),
          f"delivered {sorted(got)}; outcomes {sorted({r['Outcome'] for r in rows})}")
    spans = sorted((float(r['Start_Time']), float(r['End_Time'])) for r in rows)
    overlap = any(b[0] < a[1] for a, b in zip(spans, spans[1:]))
    check('drive: the RSU runs one transfer at a time (rsu_max_concurrent 1), the rest queue',
          not overlap and any(float(r['Queue_Wait']) > 0 for r in rows),
          f"queue waits {[float(r['Queue_Wait']) for r in rows]}")


def test_beacons():
    tmp = tempfile.mkdtemp()
    link = Ns3Link(7, RSU, NS3_DIR, '', log_path=os.path.join(tmp, 'ns3.log'))
    dists = [20, 60, 100, 150, 250, 400]
    link.set_motion({i: (RSU[0] + d, RSU[1], 0, 0) for i, d in enumerate(dists)})   # veh6 parked
    link.advance(3.0)
    heard = link.heard(0.5)
    link.close()
    near = {i for i, d in enumerate(dists) if d <= 60}
    far = {i for i, d in enumerate(dists) if d >= 400} | {6}
    check('beacons (default channel): heard close to the RSU, not far away or parked',
          near <= heard and not (far & heard),
          ', '.join(f'{d} m: {"heard" if i in heard else "-"}' for i, d in enumerate(dists)) + ', parked: '
          + ('heard' if 6 in heard else '-'))


def test_beacon_drive():
    import shared_train
    from comm import CommManager
    from dqnagent import dqnAgent
    from run_client import get_alldets
    from shared_env import SharedSumoEnv
    from utils import calculate_connections, checkSumo, get_alledges
    _, sumolib = checkSumo()
    net = 'Net/dqn.net.xml'
    edges = get_alledges(net)
    conn = calculate_connections(edges, net)
    cfg = {'use_gui': False, 'start': True, 'quit_on_end': True, 'veh_id': 0,
           'ns3': {'ns3_dir': NS3_DIR, 'fl_sim_args': '', 'discovery': 'beacon',
                   'beacon_interval': 0.1, 'beacon_timeout': 0.5, 'coverage_radius': 150},
           'pqc': {'scheme': 'ML-DSA-44', 'key_dir': 'pqc_keys'}}
    env = SharedSumoEnv(sumolib.checkBinary('sumo'), 'dqn.sumocfg', edges, get_alldets(edges), conn,
                        'E15', 94, 3, cfg)
    starts = [['E0'], ['E1'], ['E3'], ['E11']]
    slots = [shared_train.AgentSlot(f'veh{i}', dqnAgent(f'veh{i}', i, edges, conn, 94, 3, 1, '/tmp'),
                                    starts[i]) for i in range(4)]
    tmp = tempfile.mkdtemp()
    comm = CommManager(cfg, [s.veh for s in slots], tmp, RSU)
    w = [np.random.default_rng(1).normal(size=(94, 150)).astype(np.float32)]
    packets = {s.veh: comm.packet(s.veh, 1, w, 'uplink') for s in slots}
    got = comm.run_drive(env, slots, 'Rou/eval_shared_p3_scenario_0.rou.xml',
                         {s.veh: 50.0 + 10 * i for i, s in enumerate(slots)}, 600, 'uplink', 1, packets)
    comm.close()
    env.close()
    rows = list(csv.DictReader(open(os.path.join(tmp, 'comm.csv'))))
    starts_d = [float(r['Dist_Start']) for r in rows]
    check('beacon drive: transfers start only in radio range, verified deliveries returned',
          rows and max(starts_d) < 250 and set(got) == {r['Vehicle'] for r in rows if r['Outcome'] == 'delivered'},
          f'{len(rows)} attempts, start distance {min(starts_d):.0f}-{max(starts_d):.0f} m, delivered {sorted(got)}')


if __name__ == '__main__':
    test_motion_lossless_range()
    test_distance_and_determinism()
    test_signatures()
    test_beacons()
    test_beacon_drive()
    test_drive()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
