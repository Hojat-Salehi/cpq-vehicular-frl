import time
import os, sys, socket
import numpy as np
from xml.etree.ElementTree import parse
from collections import defaultdict

def _resolve_sumo_home() -> str:

    """Detect SUMO_HOME in priority order — never hardcode a path."""

    # 1. Respect what the user/shell already set

    if os.environ.get("SUMO_HOME"):

        return os.environ["SUMO_HOME"]


    # 2. Ask sumolib (works in conda envs, pip installs, etc.)

    try:

        import sumolib

        candidate = os.path.dirname(os.path.dirname(sumolib.__file__))

        if os.path.isdir(os.path.join(candidate, "tools")):

            return candidate

    except ImportError:

        pass


    # 3. Common install locations as a last resort

    for candidate in [

        "/usr/local/share/sumo",   # system-wide Linux

        "/usr/share/sumo",          # apt install on Debian/Ubuntu

        os.path.expanduser("~/sumo"),  # local user install

    ]:

        if os.path.isdir(os.path.join(candidate, "tools")):

            return candidate


    raise EnvironmentError(

        "Cannot find SUMO. Set the SUMO_HOME environment variable, e.g.:\n"

        "  export SUMO_HOME=/path/to/sumo"

    )


SUMO_HOME = _resolve_sumo_home()

os.environ["SUMO_HOME"] = SUMO_HOME   # propagate so child processes see it

tools = os.path.join(SUMO_HOME, "tools")

sys.path = [p for p in sys.path if "Eclipse\\Sumo\\tools" not in p]

if tools not in sys.path:

    sys.path.insert(0, tools) 

import traci


class dqnEnv():
    def __init__(
        self,
        sumoBinary,
        net_file: str,
        cfg_file: str,
        edgelists: list,
        alldets: list,
        dict_connection,
        veh: str,
        destination: str,
        state_size: int,
        action_size: int,
        config: dict,
        use_gui: bool = True,
        begin_time: int = 0,
        num_seconds: int = 3600,
        max_depart_delay: int = 10000,
    ):
        self.sumoBinary = sumoBinary
        self.net = net_file
        self.sumocfg = cfg_file

        self.edgelists = edgelists
        self.alldets = alldets

        self.config = config
        self.use_gui = config.get('use_gui', True)

        self.veh = veh
        self.destination = destination

        self.episode = 0
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.max_depart_delay = max_depart_delay

        self.action_size = action_size
        self.state_size = state_size

        self.dict_connection = dict_connection
        self.sumo = traci

        # runtime info (set later)
        self.agent_id = int(self.config.get('veh_id', 0))  # can be overwritten by caller
        self.port = None
        self.label = None
        self.traci_alive = False

    # -------- SUMO launch (server mode only) --------
    def _get_free_port(self):
        s = socket.socket()
        s.bind(('', 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def start_simulation(self):
        # Always run in "server" mode: launch SUMO and connect via TraCI
        if not self.traci_alive:
            self.start_simulation_server(self.agent_id)

    # def start_simulation_server(self, agent_id: int):
    #     self.agent_id = int(agent_id)
    #     self.label = f"agent_{self.agent_id}"
    #     self.port = self._get_free_port()

    #     print("Sumo binary : {}".format(self.sumoBinary))
    #     print("Sumo config : {}".format(self.sumocfg))

    #     # sumo_cmd = [
    #     #     self.sumoBinary,
    #     #     "-c", self.sumocfg,
    #     #     "--start" if self.config.get('start', True) else "",
    #     #     "--quit-on-end" if self.config.get('quit_on_end', True) else "",
    #     #     "--no-step-log", "true",
    #     #     "--time-to-teleport", "-1",
    #     # ]

    #     ################## changed here ##########################
    #     # Get agent-specific route file from config (CRITICAL FIX)
    #     agent_route_file = self.config.get('randomrou', f'Rou/dqnrandom_{agent_id}.rou.xml')
    #     print(f"Agent {agent_id} using route file: {agent_route_file}")

    #     sumo_cmd = [
    #         self.sumoBinary,
    #         "-c", self.sumocfg,
    #         "--route-files", agent_route_file,  # ← OVERRIDE route file per agent!
    #         "--start" if self.config.get('start', True) else "",
    #         "--quit-on-end" if self.config.get('quit_on_end', True) else "",
    #         "--no-step-log", "true",
    #         "--time-to-teleport", "-1",
    #     ]
    #     # #########################################################

    #     # strip empty conditionals
    #     sumo_cmd = [x for x in sumo_cmd if x]

    #     # IMPORTANT: do NOT add --remote-port here; we pass port=... to traci.start
    #     traci.start(sumo_cmd, port=self.port, label=self.label)
    #     self.sumo.setOrder(self.agent_id + 1)

    #     if self.use_gui:
    #         try:
    #             self.sumo.gui.setSchema(self.sumo.gui.DEFAULT_VIEW, "real world")
    #         except Exception:
    #             pass

    #     # Precompute edge data
    #     self.dict_edgelimits = self.get_edgelimits()
    #     destlane = self.destination + '_0'
    #     self.destCord = self.sumo.lane.getShape(destlane)[0]
    #     self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()

    #     self.traci_alive = True
    def start_simulation_server(self, agent_id: int, max_retries=3, timeout=10):
        import threading
        import datetime
        
        debug_log_file = os.path.join('log', f'veh{self.agent_id}_debug.txt')
        
        def log_debug(msg):
            # timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            # with open(debug_log_file, 'a') as f:
            #     f.write(f"[{timestamp}] {msg}\n")
            #     f.flush()
            pass
        
        self.agent_id = int(agent_id)
        self.label = f"agent_{self.agent_id}"
        
        log_debug(f"  start_simulation_server(): Entry point")
        log_debug(f"  Sumo binary: {self.sumoBinary}")
        log_debug(f"  Sumo config: {self.sumocfg}")
        
        for attempt in range(max_retries):
            try:
                import traci
                try:
                    traci.close(False)
                    log_debug("  Called traci.close(False) for cleanup")
                except Exception as e_close:
                    log_debug(f"  Warning: traci.close(False) failed (non-fatal): {e_close}")
            except Exception as e_import:
                log_debug(f"  Warning: could not import traci for cleanup: {e_import}")

            self.port = self._get_free_port()
            log_debug(f"  Attempt {attempt+1}/{max_retries}: Trying port {self.port}")
            
            # Get agent-specific route file from config
            agent_route_file = self.config.get('randomrou', f'Rou/dqnrandom_{agent_id}.rou.xml')
            log_debug(f"  Using route file: {agent_route_file}")
            
            sumo_cmd = [
                self.sumoBinary,
                "-c", self.sumocfg,
                "--route-files", agent_route_file,
                # "--seed", str(self.config.get('seed', 42)),
                "--start" if self.config.get('start', True) else "",
                "--quit-on-end" if self.config.get('quit_on_end', True) else "",
                "--no-step-log", "true",
                "--time-to-teleport", "-1",
            ]
            # strip empty conditionals
            sumo_cmd = [x for x in sumo_cmd if x]
            
            # Container to store exception from thread
            exception_container = [None]
            success_flag = [False]
            
            def start_traci():
                try:
                    log_debug(f"    Thread: About to call traci.start() on port {self.port}")
                    # traci.start(sumo_cmd, port=self.port)
                    traci.start(sumo_cmd, port=self.port, label=self.label)
                    log_debug(f"    Thread: traci.start() completed")
                    
                    log_debug(f"    Thread: Setting order to {self.agent_id + 1}")
                    self.sumo.setOrder(self.agent_id + 1)
                    
                    if self.use_gui:
                        try:
                            log_debug(f"    Thread: Setting GUI schema")
                            self.sumo.gui.setSchema(self.sumo.gui.DEFAULT_VIEW, "real world")
                        except Exception as e:
                            log_debug(f"    Thread: GUI schema failed (not critical): {e}")
                    
                    # Precompute edge data
                    log_debug(f"    Thread: Computing edge limits")
                    self.dict_edgelimits = self.get_edgelimits()
                    
                    log_debug(f"    Thread: Getting destination coordinates")
                    destlane = self.destination + '_0'
                    self.destCord = self.sumo.lane.getShape(destlane)[0]
                    
                    log_debug(f"    Thread: Getting edge lengths")
                    self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()
                    
                    log_debug(f"    Thread: All initialization complete")
                    success_flag[0] = True
                    
                except Exception as e:
                    log_debug(f"    Thread: Exception occurred: {e}")
                    exception_container[0] = e
            
            log_debug(f"  Starting thread for port {self.port}")
            thread = threading.Thread(target=start_traci)
            thread.daemon = True
            thread.start()
            
            log_debug(f"  Waiting for thread (timeout={timeout}s)...")
            thread.join(timeout=timeout)
            
            # Check if thread is still alive (timeout occurred)
            if thread.is_alive():
                log_debug(f"  TIMEOUT: Port {self.port} did not respond within {timeout}s, trying new port...")
                # Try to cleanup if possible
                try:
                    traci.close(True)
                    log_debug(f"  Cleanup: traci.close(True) called")
                except Exception as cleanup_err:
                    log_debug(f"  Cleanup failed (not critical): {cleanup_err}")
                continue
            
            # Check if exception occurred in thread
            if exception_container[0] is not None:
                log_debug(f"  ERROR on port {self.port}: {exception_container[0]}")
                if attempt == max_retries - 1:
                    log_debug(f"  All {max_retries} attempts exhausted, raising exception")
                    raise Exception(f"Failed to start SUMO after {max_retries} attempts: {exception_container[0]}")
                log_debug(f"  Retrying with new port...")
                continue
            
            # Check if success flag was set
            if not success_flag[0]:
                log_debug(f"  Thread completed but success_flag not set, something went wrong")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to start SUMO after {max_retries} attempts: Thread completed without success")
                continue
            
            # Success!
            log_debug(f"  SUCCESS: SUMO started on port {self.port}")
            self.traci_alive = True
            log_debug(f"  start_simulation_server(): Exit (success)")
            break
        else:
            # All retries exhausted
            log_debug(f"  FATAL: All {max_retries} attempts exhausted")
            raise Exception(f"[Agent {self.agent_id}] Failed to start SUMO after {max_retries} attempts with timeout {timeout}s")
    
    
    
    def sumo_step(self):
        self.sumo.simulationStep()

    # def sumoclose(self):
    #     if self.traci_alive:
    #         try:
    #             traci.close(True)  # don't kill the process, just close TraCI
    #         finally:
    #             self.traci_alive = False
    def sumoclose(self):
        import datetime
        debug_log_file = os.path.join('log', f'{self.veh}_debug.txt')
        def log_debug(msg):
            # timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            # # if needed, uncomment below to log to file
            # with open(debug_log_file, 'a') as f:
            #     f.write(f"[{timestamp}] {msg}\n")
            #     f.flush()
            pass
        if self.traci_alive:
            try:
                log_debug(f"  sumoclose(): Closing SUMO for agent {self.agent_id}")
                traci.close(True)  # Force kill SUMO process
                log_debug(f"  sumoclose(): SUMO closed successfully")
            except Exception as e:
                log_debug(f"  sumoclose(): Error closing SUMO: {e}")
            finally:
                self.traci_alive = False
        else:
            log_debug(f"  sumoclose(): SUMO already closed (traci_alive=False)")

    # def reset(self):
    #     self.episode += 1
    #     self.start_simulation()

    #     # vehicle create
    #     route_id = "rou" + self.veh[3:]
    #     id_ = int(self.veh[3:])
    #     route = self.config['start_edges']
    #     route = route[id_]
    #     self.sumo.route.add(route_id, route)  # default route
    #     self.sumo.vehicle.add(self.veh, route_id, departLane='best')

    #     curlane = self.get_curlane(self.veh)
    #     while curlane == '':
    #         curlane = self.get_curlane(self.veh)
    #         self.sumo.simulationStep()

    #     curedge = self.get_curedge(curlane)
    #     state = self.get_state(self.veh, curedge)
    #     return state


    def reset(self):
        import datetime
        debug_log_file = os.path.join('log', f'{self.veh}_debug.txt')
        def log_debug(msg):
            # timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            # # if needed, uncomment below to log to file
            # with open(debug_log_file, 'a') as f:
            #     f.write(f"[{timestamp}] {msg}\n")
            #     f.flush()
            pass

        log_debug(f"  reset(): Entry point")
        # NEW: Force close any existing SUMO before starting new episode
        log_debug(f"  reset(): Force closing any existing SUMO instance")
        self.sumoclose()  # ← ADD THIS
        
        self.episode += 1
        
        log_debug(f"  reset(): About to call start_simulation()")
        self.start_simulation()
        log_debug(f"  reset(): start_simulation() completed")

        # vehicle create
        route_id = "rou" + self.veh[3:]
        id_ = int(self.veh[3:])
        route = self.config['start_edges']
        route = route[id_]
        
        log_debug(f"  reset(): About to add route {route_id} with edges {route}")
        self.sumo.route.add(route_id, route)
        log_debug(f"  reset(): Route added")
        
        log_debug(f"  reset(): About to add vehicle {self.veh}")
        self.sumo.vehicle.add(self.veh, route_id, departLane='best')
        log_debug(f"  reset(): Vehicle added")

        log_debug(f"  reset(): About to get current lane")
        curlane = self.get_curlane(self.veh)
        log_debug(f"  reset(): Initial curlane = '{curlane}'")
        
        log_debug(f"  reset(): Entering while loop to wait for vehicle")
        while curlane == '':
            curlane = self.get_curlane(self.veh)
            self.sumo.simulationStep()

        log_debug(f"  reset(): Exited while loop, curlane = '{curlane}'")
        
        curedge = self.get_curedge(curlane)
        log_debug(f"  reset(): Got curedge = '{curedge}'")
        
        state = self.get_state(self.veh, curedge)
        log_debug(f"  reset(): Got state, returning")
        
        return state
    # -------- helpers --------
    def get_curlane(self, veh):
        return self.sumo.vehicle.getLaneID(veh)

    def get_curedge(self, curlane):
        return self.sumo.lane.getEdgeID(curlane)

    def get_done(self, curedge):
        return curedge == self.destination

    def get_RoadID(self, veh):
        return traci.vehicle.getRoadID(veh)

    def get_reward(self, curedge, nextedge):
        det = nextedge.replace('E', 'D')
        num_veh = self.get_numVeh(det)
        if num_veh == 1:
            traveltime = self.dict_edgelengths[nextedge] / self.dict_edgelimits[nextedge]
        else:
            traveltime = self.sumo.edge.getTraveltime(nextedge)
        return -traveltime
    def get_nextedge(self, curedge, action):
        return self.dict_connection[curedge][action]

    def get_numVeh(self, det):
        return self.sumo.lanearea.getLastStepVehicleNumber(det)

    def get_edgelengths(self):
        dict_edgelengths = defaultdict(float)
        list_edgelengths = []
        dict_edgelengths.update((k, 0.0) for k in self.edgelists)
        for edge in self.edgelists:
            lane = edge + '_0'
            length = self.sumo.lane.getLength(lane)
            dict_edgelengths[edge] = length
            list_edgelengths.append(length)
        return dict_edgelengths, list_edgelengths

    def get_edgelimits(self):
        dict_edgelimits = defaultdict(float)
        dict_edgelimits.update((k, 15.0) for k in self.edgelists)
        dict_edgelimits['E7'] = 10.0
        dict_edgelimits['-E7'] = 10.0
        return dict_edgelimits

    def get_state(self, veh, curedge):  # total 94 entries
        state = []
        # 1) vehicles per edge (20)
        for edge in self.edgelists:
            det = edge.replace("E", "D")
            state.append(self.get_numVeh(det))
        # 2) mean speed per edge (20)
        for edge in self.edgelists:
            state.append(self.sumo.edge.getLastStepMeanSpeed(edge))
        # 3) edge lengths (20)
        state.extend(self.list_edgelengths)
        # 4) current coords (2)
        curlane = curedge + '_0'
        curCord = self.sumo.lane.getShape(curlane)[0]
        state.extend(list(curCord))
        # 5) destination coords (2)
        state.extend(list(self.destCord))
        return state

    def get_nextstate(self, veh, nextedge):
        next_state = []
        # 1) vehicles per edge (20)
        for edge in self.edgelists:
            det = edge.replace("E", "D")
            next_state.append(self.get_numVeh(det))
        # 2) mean speed per edge (20)  (fixed: call function)
        for edge in self.edgelists:
            next_state.append(self.sumo.edge.getLastStepMeanSpeed(edge))
        # 3) edge lengths (20) (fixed: extend, don't append the list)
        next_state.extend(self.list_edgelengths)
        # 4) next coords (2)
        nextlane = nextedge + '_0'
        nextCord = self.sumo.lane.getShape(nextlane)[0]
        next_state.extend(list(nextCord))
        # 5) destination coords (2)
        next_state.extend(list(self.destCord))
        return next_state

    # def step(self, curedge, nextedge):
    #     beforeedge = curedge
    #     done = self.get_done(curedge)
    #     reward = self.get_reward(curedge, nextedge)

    #     if done:
    #         return reward, done

    #     self.sumo.vehicle.changeTarget(self.veh, nextedge)

    #     while self.sumo.simulation.getMinExpectedNumber() > 0:
    #         curedge = self.get_RoadID(self.veh)
    #         done = self.get_done(curedge)
    #         if done:
    #             break
    #         self.sumo.simulationStep()
    #         if curedge in self.edgelists and curedge != beforeedge:
    #             break

    #     return reward, done
    def step(self, curedge, nextedge):
        import datetime
        debug_log_file = os.path.join('log', f'{self.veh}_debug.txt')
        
        def log_debug(msg):
            # timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            # # if needed, uncomment below to log to file
            # with open(debug_log_file, 'a') as f:
            #     f.write(f"[{timestamp}] {msg}\n")
            #     f.flush()
            pass
        
        beforeedge = curedge
        done = self.get_done(curedge)
        reward = self.get_reward(curedge, nextedge)
        
        log_debug(f"  step() START: {curedge} -> {nextedge}, reward={reward:.2f}, done={done}")

        if done:
            log_debug(f"  step() RETURN early (done=True)")
            return reward, done

        log_debug(f"  About to call changeTarget({nextedge})")
        self.sumo.vehicle.changeTarget(self.veh, nextedge)
        log_debug(f"  changeTarget returned successfully")
        
        # Check if vehicle still exists after changeTarget
        try:
            log_debug(f"  Checking if vehicle still exists in simulation...")
            veh_list = self.sumo.vehicle.getIDList()
            log_debug(f"  Vehicle list has {len(veh_list)} vehicles: {veh_list}")
            if self.veh not in veh_list:
                log_debug(f"  ERROR: Our vehicle {self.veh} NOT in vehicle list!")
        except Exception as e:
            log_debug(f"  ERROR checking vehicle list: {e}")
        
        step_counter = 0
        log_debug(f"  About to call getMinExpectedNumber...")
        minExpected = self.sumo.simulation.getMinExpectedNumber()
        log_debug(f"  getMinExpectedNumber returned: {minExpected}")
        log_debug(f"  Entering while loop")
        MAX_STEPS = 1000
        while self.sumo.simulation.getMinExpectedNumber() > 0:
            step_counter += 1
            # VALIDATION: Break if stuck too long
            if step_counter > MAX_STEPS:
                log_debug(f"  step() TIMEOUT: Stuck for {step_counter} steps, breaking")
                done = True
                break
            
            log_debug(f"  === Loop iteration {step_counter} START ===")
            
            log_debug(f"    About to call get_RoadID({self.veh})...")
            try:
                curedge = self.get_RoadID(self.veh)
                log_debug(f"    get_RoadID returned: '{curedge}'")
            except Exception as e:
                log_debug(f"    ERROR in get_RoadID: {e}")
                break
            
            log_debug(f"    About to call get_done('{curedge}')...")
            done = self.get_done(curedge)
            log_debug(f"    get_done returned: {done}")
            
            if done:
                log_debug(f"    Destination reached! Breaking.")
                break
            
            try:
                sim_time = self.sumo.simulation.getTime()
                num_vehicles = len(self.sumo.vehicle.getIDList())
                log_debug(f"    Before simulationStep: sim_time={sim_time}, num_vehicles={num_vehicles}, minExpected={self.sumo.simulation.getMinExpectedNumber()}")
            except Exception as e:
                log_debug(f"    ERROR before simulationStep: {e}")
            
            log_debug(f"    About to call simulationStep()...")
            self.sumo.simulationStep()
            log_debug(f"    simulationStep completed")
            
            log_debug(f"    Checking break condition: curedge='{curedge}', beforeedge='{beforeedge}'")
            log_debug(f"    Is '{curedge}' in edgelists? {curedge in self.edgelists}")
            log_debug(f"    Is '{curedge}' != '{beforeedge}'? {curedge != beforeedge}")
            
            if curedge in self.edgelists and curedge != beforeedge:
                log_debug(f"    Moved to new edge! Breaking.")
                break
            
            log_debug(f"  === Loop iteration {step_counter} END ===")
            
            # Safety check every 10 iterations
            if step_counter % 10 == 0:
                minExp = self.sumo.simulation.getMinExpectedNumber()
                log_debug(f"  [Checkpoint] Iteration {step_counter}, minExpected={minExp}")
        
        log_debug(f"  Exited while loop after {step_counter} iterations")
        log_debug(f"  step() RETURN: reward={reward:.2f}, done={done}")
        return reward, done
