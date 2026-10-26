# shared_env.py
# One SUMO instance shared by all agent vehicles (design change 1).
#
# dqnEnv owns the simulation for a single vehicle: its step() advances SUMO
# until that vehicle changes edge. Here SUMO is stepped centrally and every
# agent vehicle reports when it reaches a decision point. The state comes
# from dqnEnv unchanged.

import os

import numpy as np

from dqnenv import dqnEnv, traci


class SharedSumoEnv(dqnEnv):
    LABEL = 'shared'
    EMPTY_ROUTES = 'Rou/empty.rou.xml'

    def __init__(self, sumoBinary, cfg_file, edgelists, alldets, dict_connection,
                 destination, state_size, action_size, config):
        super().__init__(
            sumoBinary, net_file=None, cfg_file=cfg_file, edgelists=edgelists,
            alldets=alldets, dict_connection=dict_connection, veh=None,
            destination=destination, state_size=state_size,
            action_size=action_size, config=dict(config, veh_id=0),
        )
        self._edgeset = set(edgelists)
        self._ff_time = {}
        self.encoding = config.get('state_encoding', 'raw')
        self._edge_index = {e: i for i, e in enumerate(edgelists)}

    # -------- lifecycle --------
    def start(self, route_file, tripinfo=None):
        """(Re)start the simulation with the given background traffic.

        The SUMO process is launched once and reused: later episodes call
        traci.load, which avoids TraCI's 1 s connect retry on every launch.
        tripinfo: optional path for SUMO's per-vehicle trip summary (including
        vehicles still driving when the simulation is reloaded or closed; the
        file is complete only after the next reload + step, or after close()).
        """
        sumo_cmd = [
            self.sumoBinary, '-c', self.sumocfg,
            '--route-files', route_file,
            '--no-step-log', 'true',
            '--time-to-teleport', '-1',
        ]
        if tripinfo:
            sumo_cmd += ['--tripinfo-output', tripinfo, '--tripinfo-output.write-unfinished', 'true']
        if self.config.get('start', True):
            sumo_cmd.append('--start')
        if self.config.get('quit_on_end', True):
            sumo_cmd.append('--quit-on-end')
        try:
            traci.switch(self.LABEL)          # a SUMO process for this label is running
            traci.load(sumo_cmd[1:])
        except traci.exceptions.TraCIException:
            traci.start(sumo_cmd, label=self.LABEL)
        self.traci_alive = True
        self._traffic = None

        self.dict_edgelimits = self.get_edgelimits()
        self.destCord = self.sumo.lane.getShape(self.destination + '_0')[0]
        self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()
        self._lane_start = {}

    def close(self):
        """Stop the SUMO process."""
        try:
            traci.switch(self.LABEL)
            traci.close(True)
        except Exception:
            pass
        self.traci_alive = False

    # -------- vehicles --------
    def add_vehicle(self, veh, start_edges, depart=0.0):
        """Add an agent vehicle that departs at `depart` (s; inserted as soon as possible after)."""
        route_id = 'rou_' + veh
        self.sumo.route.add(route_id, list(start_edges))
        self.sumo.vehicle.add(veh, route_id, depart=f'{depart:g}', departLane='best')
        # No random speed factor: travel times differ only through routing and traffic
        self.sumo.vehicle.setSpeedFactor(veh, 1.0)

    def remove_vehicle(self, veh):
        try:
            self.sumo.vehicle.remove(veh)
        except traci.exceptions.TraCIException:
            pass  # already arrived / not inserted
        self._traffic = None   # removal changes edge counts/speeds within the current step

    def road(self, veh):
        """Current road id, '' before insertion, None once the vehicle has left."""
        try:
            return self.sumo.vehicle.getRoadID(veh)
        except traci.exceptions.TraCIException:
            return None

    def is_edge(self, road):
        return road in self._edgeset

    def change_target(self, veh, nextedge):
        self.sumo.vehicle.changeTarget(veh, nextedge)

    def valid_actions(self, curedge):
        return [i for i, e in enumerate(self.dict_connection[curedge]) if e != '']

    def time(self):
        return self.sumo.simulation.getTime()

    def delta_t(self):
        return self.sumo.simulation.getDeltaT()

    def speed(self, veh):
        return self.sumo.vehicle.getSpeed(veh)

    def free_flow_edge_time(self, edge):
        """Edge length / speed limit (s)."""
        if edge not in self._ff_time:
            lane = edge + '_0'
            self._ff_time[edge] = self.sumo.lane.getLength(lane) / self.sumo.lane.getMaxSpeed(lane)
        return self._ff_time[edge]

    def free_flow_speed(self, edge):
        return self.sumo.lane.getLength(edge + '_0') / self.free_flow_edge_time(edge)

    def current_edge_time(self, edge):
        """SUMO's current travel time estimate for an edge (length / last-step mean speed)."""
        return self.sumo.edge.getTraveltime(edge)

    def trip_stats(self, veh):
        """(distance driven [m], SUMO time loss [s]) of a vehicle still in the network."""
        try:
            return self.sumo.vehicle.getDistance(veh), self.sumo.vehicle.getTimeLoss(veh)
        except traci.exceptions.TraCIException:
            return float('nan'), float('nan')

    def free_flow_times(self, start_edges):
        """Travel time of one vehicle alone in the empty network, from each start edge
        to entering the destination (same start/end definition as the agents' Travel_Time)."""
        if not os.path.exists(self.EMPTY_ROUTES):
            with open(self.EMPTY_ROUTES, 'w') as f:
                f.write('<routes/>\n')
        out = {}
        for k, start in enumerate(dict.fromkeys(e[0] for e in start_edges)):
            self.start(self.EMPTY_ROUTES)
            veh = f'freeflow{k}'
            self.add_vehicle(veh, self.sumo.simulation.findRoute(start, self.destination).edges)
            depart = None
            while True:
                road = self.road(veh)
                if depart is None and road:
                    depart = self.time()
                if road == self.destination or road is None:
                    out[start] = self.time() - depart
                    break
                self.step()
        return out

    def step(self):
        self.sumo.simulationStep()
        self._traffic = None

    def positions(self, vehs):
        """{veh: (x, y)} for vehicles currently in the network (for ns-3 coupling)."""
        out = {}
        for v in vehs:
            try:
                out[v] = tuple(self.sumo.vehicle.getPosition(v))
            except traci.exceptions.TraCIException:
                pass
        return out

    @staticmethod
    def state_dim(encoding, num_edges):
        """'raw': dqnEnv.get_state layout (counts, speeds, edge lengths, current and
        destination coordinates: 3n + 4). 'onehot': current edge one-hot, occupancy
        and relative speed per edge (3n)."""
        return 3 * num_edges + (4 if encoding == 'raw' else 0)

    def motion(self, vehs):
        """{veh: (x, y, vx, vy)} in SUMO coordinates (m, m/s) for vehicles in the network."""
        import math
        out = {}
        for v in vehs:
            try:
                x, y = self.sumo.vehicle.getPosition(v)
                speed = self.sumo.vehicle.getSpeed(v)
                angle = math.radians(self.sumo.vehicle.getAngle(v))   # 0 = north, clockwise
            except traci.exceptions.TraCIException:
                continue
            out[v] = (x, y, speed * math.sin(angle), speed * math.cos(angle))
        return out

    def junction_position(self, junction):
        return tuple(self.sumo.junction.getPosition(junction))

    def get_state_np(self, curedge):
        """State of an agent on curedge. The traffic part is read once per simulation
        step and shared by all agents deciding in that step.

        'raw' (default) equals dqnEnv.get_state. 'onehot' encodes the current edge
        as one-hot and normalizes traffic: vehicles / edge capacity (7.5 m per
        vehicle) and mean speed / speed limit; constant inputs are left out."""
        if self._traffic is None:
            counts = [self.get_numVeh(e.replace('E', 'D')) for e in self.edgelists]
            speeds = [self.sumo.edge.getLastStepMeanSpeed(e) for e in self.edgelists]
            self._traffic = counts + speeds
        if self.encoding == 'onehot':
            n = len(self.edgelists)
            onehot = np.zeros(n, dtype=np.float32)
            onehot[self._edge_index[curedge]] = 1.0
            occupancy = [c * 7.5 / self.dict_edgelengths[e]
                         for c, e in zip(self._traffic[:n], self.edgelists)]
            rel_speed = [v / self.free_flow_speed(e)
                         for v, e in zip(self._traffic[n:], self.edgelists)]
            return np.concatenate([onehot, np.asarray(occupancy + rel_speed, dtype=np.float32)])[None, :]
        if curedge not in self._lane_start:
            self._lane_start[curedge] = list(self.sumo.lane.getShape(curedge + '_0')[0])
        state = self._traffic + self.list_edgelengths + self._lane_start[curedge] + list(self.destCord)
        return np.asarray(state, dtype=np.float32)[None, :]
