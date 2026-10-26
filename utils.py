import yaml
import os
import sys
import logging
from xml.etree.ElementTree import parse
from collections import defaultdict
import xml.etree.ElementTree as ET
import csv 
import datetime
import random

# function to equire run id
def require_run_id() -> str:
    rid = os.environ.get("FL_RUN_ID", "").strip()
    if not rid:
        import datetime
        rid = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.environ["FL_RUN_ID"] = rid
    return rid


def client_veh_id(client, cache) -> str:
    """Vehicle id ('vehK') of a Flower client, asked once via get_properties and cached by cid.

    The server must use the client's own id: ns-3 downlink ports and the PQC
    uplink files are keyed by it.
    """
    cid = str(client.cid)
    if cid not in cache:
        import flwr as fl
        res = client.get_properties(fl.common.GetPropertiesIns(config={}), timeout=60, group_id=None)
        cache[cid] = str(res.properties['veh'])
    return cache[cid]


def ns3_drop_undelivered() -> bool:
    """True when updates whose ns-3 uplink failed must be left out of aggregation."""
    return os.environ.get("FL_NS3_DROP_UNDELIVERED", "1") == "1"

def setup_csv(out_dir,veh_id,trained):
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    current_datetime = datetime.datetime.now()
    current_time = current_datetime.time()
    current_time = str(current_datetime).split('.')[0].replace(':','-').replace(' ','_')
    if not trained:
        file_name = f'{current_time}_{veh_id}_train.csv'
    else:
        file_name = f'{current_time}_{veh_id}_test.csv'
    file_path = os.path.join(out_dir,file_name)
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode','Waiting_Time','Total_Time','simu_time'])
    return file_path

def setup_csv_stable(out_dir, veh_id, trained):
    """Create a CSV with a stable (non-timestamped) name for logical clients.
    Safe to call multiple times — if the file already exists it is left intact
    so data is not wiped on reconnect. Columns: shared_train.CSV_HEADER.
    """
    from shared_train import CSV_HEADER
    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    suffix = trained if isinstance(trained, str) else ('train' if not trained else 'test')
    file_name = f'{veh_id}_{suffix}.csv'
    file_path = os.path.join(out_dir, file_name)
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADER)
    return file_path


def setup_eval_csv(out_dir, method, run_id):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, 'global_eval.csv')
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        from shared_train import EVAL_SUMMARY_HEADER
        writer.writerow(EVAL_SUMMARY_HEADER)
    return file_path


def save_data(file_path,data):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


def setUpLogger():
    log_level = logging.WARN
    logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)d %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("Logger started")
    return logger

def parseConfig(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)
            p_config = {}
            # ── existing keys ──
            p_config['veh_id']        = getattr(args, 'vehicle', None)
            p_config['plot_results']  = config['common']['plot_results']
            p_config['cfg_file']      = config['common']['cfg_file']
            p_config['total_episodes']= config['common']['total_episodes']
            p_config['port']          = config['common']['port']
            p_config['server_ip']     = config['common']['server_ip']
            p_config['num_clients']   = config['sumo']['num_clients']   # single source
            p_config['use_gui']       = config['sumo']['use_gui']
            p_config['start']         = config['sumo']['start']
            # p_config['start_edges']   = config['sumo']['start_edges']
            p_config['quit_on_end']   = config['sumo']['quit_on_end']
            p_config['shared_env']    = bool(config['sumo'].get('shared_env', False))
            p_config['background_period'] = float(config['sumo'].get('background_period', 3))
            p_config['warmup']        = float(config['sumo'].get('warmup', 0))
            p_config['depart_window'] = float(config['sumo'].get('depart_window', 0))
            _base_edges = config['sumo'].get('start_edges', [['E0'], ['E1'], ['E3'], ['E11'], ['E8']])
            _n          = p_config['num_clients']
            p_config['start_edges'] = [_base_edges[i % len(_base_edges)] for i in range(_n)]
            # ── new experiment keys ──
            exp = config.get('experiment', {})
            p_config['method']           = exp.get('method', 'none')
            p_config['true_rounds']      = int(exp.get('true_rounds', 50))
            p_config['batches_per_agg']  = int(exp.get('batches_per_agg', 1))
            p_config['num_rounds']       = p_config['true_rounds'] * p_config['batches_per_agg']
            p_config['out_dir']          = os.environ.get('FL_OUT_DIR') or exp.get('out_dir', 'results')
            p_config['epsilon_start']    = float(exp.get('epsilon_start', 1.0))
            p_config['epsilon_end']      = float(exp.get('epsilon_end', 0.05))
            p_config['epsilon_schedule'] = str(exp.get('epsilon_schedule', 'exponential'))
            p_config['epsilon_decay_rounds'] = int(exp.get('epsilon_decay_rounds', p_config['true_rounds']))
            p_config['eval_episodes']    = int(exp.get('eval_episodes', 3))
            p_config['max_trip_time']    = float(exp.get('max_trip_time', 600))
            p_config['state_encoding']   = str(exp.get('state_encoding', 'raw'))
            p_config['background_metrics'] = bool(exp.get('background_metrics', True))
            p_config['road_log']         = bool(exp.get('road_log', False))
            p_config['seed']             = int(os.environ.get('FL_SEED') or exp.get('seeds', [1])[0])
            p_config['target_update_every'] = int(exp.get('target_update_every', 1000))  # DQN target copy period (gradient steps)
            p_config['mask_target']      = bool(exp.get('mask_target', False))   # DQN target max over valid next turns only
            # ── privacy keys ──
            priv = config.get('privacy', {})
            p_config['epsilon']       = float(priv.get('epsilon', 1.0))
            p_config['delta']         = float(priv.get('delta', 1e-5))
            p_config['lambda_smooth'] = float(priv.get('lambda_smooth', 1.0))
            p_config['clip_radius']   = priv.get('clip_radius')   # cpq / ldpfl: per-layer radius
            p_config['cpq_lambda']    = float(priv.get('cpq_lambda', 1.0))   # 1 = CPQ, <1 = CPQ-lambda
            p_config['cpq_pairing_granularity'] = str(priv.get('cpq_pairing_granularity', 'client'))
            p_config['cpq_pairing']   = str(priv.get('cpq_pairing', 'server'))       # server | local
            p_config['cpq_group_key'] = str(priv.get('cpq_group_key', 'seeded'))     # local only
            if p_config['cpq_pairing'] not in ('server', 'local'):
                raise ValueError("privacy.cpq_pairing must be 'server' or 'local'")
            if p_config['cpq_group_key'] not in ('seeded', 'ml_kem'):
                raise ValueError("privacy.cpq_group_key must be 'seeded' or 'ml_kem'")
            if p_config['method'] in ('cpq', 'ldpfl', 'gaussian', 'laplace') and p_config['clip_radius'] is None:
                raise ValueError("privacy.clip_radius is required for the privacy methods")
            # ── ns3 keys ──
            ns3 = config.get('ns3', {})
            p_config['use_ns3']       = bool(ns3.get('use_ns3', False))
            p_config['ns3'] = {
                'ns3_dir'        : ns3.get('ns3_dir', '~/ns-3.46'),
                'fl_sim_args'    : ns3.get('fl_sim_args', '') or '',
                'rsu_junction'   : str(ns3.get('rsu_junction', 'J3')),
                'coverage_radius': float(ns3.get('coverage_radius', 150)),
                'rsu_max_concurrent': int(ns3.get('rsu_max_concurrent', 1)),
                'discovery'      : str(ns3.get('discovery', 'beacon')),
                'beacon_interval': float(ns3.get('beacon_interval', 0.1)),
                'beacon_timeout' : float(ns3.get('beacon_timeout', 0.5)),
                'beacon_min'     : int(ns3.get('beacon_min', 3)),
                'setup_max_drives': int(ns3.get('setup_max_drives', 20)),
                'schedule_step'  : float(ns3.get('schedule_step', 1.0)),
            }
            p_config['pqc'] = {
                'enabled'  : p_config['use_ns3'],
                'scheme'   : ns3.get('pqc_scheme', 'ML-DSA-44'),
                'key_dir'  : ns3.get('key_dir', 'pqc_keys'),
            }
            p_config['oracle'] = {
                'enabled'  : p_config['use_ns3'],
                'ip'       : ns3.get('oracle_ip', '127.0.0.1'),
                'port'     : ns3.get('oracle_port', 12345),
                'downlink' : {
                    'enabled'   : p_config['use_ns3'],
                    'listen_host': '127.0.0.1',
                    'base_port' : ns3.get('downlink_base_port', 25000),
                }
            }
            return p_config
        except yaml.YAMLError as exc:
            print(exc)

def checkSumo():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'],'tools')
        sys.path.append(tools)
        try:
            import traci
            import sumolib
            return traci, sumolib
        except ImportError:
            raise EnvironmentError("Declare SUMO_HOME environment")
    else:
        sys.exit('Declare environment variable "SUMO_HOME"')

def get_toedges(net, fromedge):
    #calculate reachable nextedges
    tree = parse(net)
    root = tree.getroot()
    toedges = []
    for connection in root.iter("connection"):
        if connection.get("from")==fromedge:
            toedges.append(connection.get("to"))
    return toedges

def get_alledges(net):
    #get plain edges by parsing net.xml
    tree = parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    edgelists = [edge.get("id") for edge in alledgelists if ':' not in edge.get("id")]
    return edgelists

def generate_detectionfile(net, det):
    #generate det.xml file by setting a detector at the end of each lane (-10m)
    with open(det,"w") as detections:
        alledges = get_alledges(net)
        alllanes = [edge +'_0' for edge in alledges]
        alldets =  [edge.replace("E","D") for edge in alledges]
        
        pos = -10
        print('<additional>', file = detections)
        for i in range(len(alledges)):
            print(' <e1Detector id="%s" lane="%s" pos="%i" freq="30" file="cross.out" friendlyPos="x"/>'
            %(alldets[i], alllanes[i],pos), file = detections)
        print('</additional>', file = detections)
        return alldets

def calculate_connections(edgelists, net):
    # calculate dictionary of reachable edges(next edge) for every edge  
    tree = parse(net)
    root = tree.getroot()
    
    dict_connection = defaultdict(list)
    dict_connection.update((k,[]) for k in edgelists)

    for connection in root.iter("connection"):
        curedge = connection.get("from")
        if ':' not in curedge:
            dict_connection[curedge].append(connection.get("to"))

    for k,v in dict_connection.items():
        if len(v)==0:
            dict_connection[k]=['','','']
        elif len(v)==1:
            dict_connection[k].append('')
            dict_connection[k].append('')
        elif len(v)==2:
            dict_connection[k].append('')
    return dict_connection 


def getRandomRoute(dictconnection,sumo):
    start_edge = 0
    dest_edge = 0
    valid_connections = {}
    # print(dictconnection)
    while ("ON A PAS UNE ROUTE VALIDE"):
        while (start_edge == dest_edge):
            start_edge = random.randint(0, len(dictconnection)-1)
            dest_edge = random.randint(0, len(dictconnection)-1)
        i = 0
        j = 0
        for it in dictconnection:
            if(i == start_edge):
                pos = it
                j += 1
            if(i == dest_edge):
                dest = it
                j += 1
            if(j == 2):
                break
            i += 1
            if(i > len(dictconnection)):
                i = 0
        route = sumo.simulation.findRoute(pos, dest)
        nodes = sumo.simulation.findRoute(pos, dest).edges
        if (len(nodes) > 3):
            print("out")
            break
        else:
            start_edge = 0
            dest_edge = 0
    return nodes


# def plot_result(episodenum, lst_cntSuccess):
#     ax = plt.figure().gca()
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.xticks(range(episodenum),rotation=45)
#     plt.plot(lst_cntSuccess, 'r-')
    
#     plt.savefig('./Qlearning/Result/qlresult%i.png' %episodenum )