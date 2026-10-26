# ns3_link.py
# Driver for fl-sim in lockstep mode (fl-sim.cc, --lockstep=true): ns-3 follows SUMO's
# simulation time and vehicle positions instead of running in real time.
#
#   link = Ns3Link(num_vehicles=10, rsu=(300, 100), fl_sim_args='')
#   link.set_motion({0: (x, y, vx, vy)}); link.park([3])
#   msg = link.send_update('veh0', round, packet)          # starts now, returns a message id
#   for ev in link.advance(t): ...                         # run ns-3 to t, collect completions
#   link.close()
#
# Events: Ns3Event(direction 'UL'|'DL', msg_id, veh_index, ok, t, payload bytes).

import glob
import os
import socket
import subprocess
import time
from collections import namedtuple

Ns3Event = namedtuple('Ns3Event', 'direction msg_id veh ok t payload')


def find_fl_sim(ns3_dir):
    hits = sorted(glob.glob(os.path.join(os.path.expanduser(ns3_dir), 'build', 'scratch',
                                         'ns3.*-fl-sim-*')))
    if not hits:
        raise FileNotFoundError(f'fl-sim binary not found under {ns3_dir}/build/scratch '
                                '(see ns3_integration/README_NS3.md)')
    return hits[0]


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Ns3Link:
    def __init__(self, num_vehicles, rsu, ns3_dir='~/ns-3.46', fl_sim_args='', log_path=None,
                 run_id=None, beacon_interval=0.1):
        self.port = _free_port()
        cmd = [find_fl_sim(ns3_dir), f'--numVehicles={num_vehicles}', '--lockstep=true',
               f'--rsuX={rsu[0]}', f'--rsuY={rsu[1]}', f'--controlPort={self.port}',
               '--logMobility=false', f'--beaconInterval={beacon_interval}'] + fl_sim_args.split()
        env = dict(os.environ)
        if run_id:
            env['FL_RUN_ID'] = run_id
        self._log = open(log_path, 'w') if log_path else subprocess.DEVNULL
        # fl-sim writes pqc_logs/<run id>/ns3_wave_metrics.csv relative to its working directory
        cwd = os.path.dirname(os.path.abspath(log_path)) if log_path else None
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        self.sock = None
        deadline = time.time() + 60
        while self.sock is None:
            if self.proc.poll() is not None:
                raise RuntimeError(f'fl-sim exited early (code {self.proc.returncode}); see {log_path}')
            try:
                self.sock = socket.create_connection(('127.0.0.1', self.port), timeout=600)
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b''

    # ── protocol ────────────────────────────────────────────────────────────────
    def _send(self, line, payload=b''):
        self.sock.sendall(line.encode('ascii') + b'\n' + payload)

    def _readline(self):
        while b'\n' not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError('fl-sim closed the connection')
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        return line.decode('ascii').strip()

    def _readexact(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self._buf)))
            if not chunk:
                raise ConnectionError('fl-sim closed the connection')
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _expect(self, reply):
        got = self._readline()
        if got != reply:
            raise RuntimeError(f'fl-sim replied {got!r}, expected {reply!r}')

    # ── commands ────────────────────────────────────────────────────────────────
    def set_motion(self, motion):
        """motion: {vehicle index: (x, y, vx, vy)} in SUMO coordinates (m, m/s)."""
        if motion:
            parts = ' '.join(f'{i} {x:.3f} {y:.3f} {vx:.3f} {vy:.3f}'
                             for i, (x, y, vx, vy) in motion.items())
            self._send('POS ' + parts)
            self._expect('OK')

    def park(self, indices):
        """Move vehicles out of range (not in the network)."""
        if indices:
            self._send('PARK ' + ' '.join(str(i) for i in indices))
            self._expect('OK')

    def _start(self, kind, veh_id, round_num, packet):
        self._send(f'{kind} {veh_id} {int(round_num)} {len(packet)}', packet)
        reply = self._readline()
        if not reply.startswith('MSG '):
            raise RuntimeError(f'fl-sim {kind} failed: {reply}')
        return int(reply.split()[1])

    def send_update(self, veh_id, round_num, packet):
        """Start an uplink (vehicle -> RSU) at the current ns-3 time; returns the message id."""
        return self._start('SEND_UPDATE', veh_id, round_num, packet)

    def send_global(self, veh_id, round_num, packet):
        """Start a downlink (RSU -> vehicle) at the current ns-3 time; returns the message id."""
        return self._start('SEND_GLOBAL', veh_id, round_num, packet)

    def advance(self, t):
        """Run ns-3 until simulation time t (s); returns the transfers completed meanwhile."""
        self._send(f'ADVANCE {t:.6f}')
        header = self._readline().split()
        if header[0] != 'EVENTS':
            raise RuntimeError(f'fl-sim ADVANCE failed: {header}')
        events = []
        for _ in range(int(header[1])):
            d, msg, veh, ok, te, n = self._readline().split()
            payload = self._readexact(int(n)) if int(n) else b''
            events.append(Ns3Event(d, int(msg), int(veh), ok == '1', float(te), payload))
        return events

    def heard(self, window, min_count=1):
        """Vehicle indices that received at least min_count RSU beacons in the last `window` s."""
        self._send(f'HEARD {window:.6f} {int(min_count)}')
        reply = self._readline().split()
        return set(int(i) for i in reply[2:])

    def status(self, index):
        """(x, y, distance to the RSU) of a vehicle as ns-3 sees it."""
        self._send(f'STATUS {index}')
        x, y, d = self._readline().split()
        return float(x), float(y), float(d)

    def now(self):
        self._send('NOW')
        return float(self._readline())

    def reset(self):
        """Drop all transfers in flight and park every vehicle."""
        self._send('RESET')
        self._expect('OK')

    def close(self):
        if self.sock is not None:
            try:
                self._send('QUIT')
                self._readline()
            except OSError:
                pass
            self.sock.close()
            self.sock = None
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self._log is not subprocess.DEVNULL:
            self._log.close()
