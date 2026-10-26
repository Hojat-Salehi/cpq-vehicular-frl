"""Launch the runs of one experiment of the paper.

    python tools/run_experiment.py E1 --dry-run   # list the runs and their state
    python tools/run_experiment.py E1             # run them (3 in parallel)
    python tools/run_experiment.py E1 --eps 1     # only the eps = 1 runs

Each run is configs/base.yaml plus its changes; configs, run outputs and logs go to
experiments/final/<folder>/. Finished runs are skipped, so the command can be repeated.
"""
import argparse
import copy
import csv
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import yaml

ROOT = os.path.join('experiments', 'final')
BASE = os.path.join('configs', 'base.yaml')
METHODS = ['cpq', 'ldpfl', 'gaussian', 'laplace']
EPS = (0.3, 0.5, 1, 2, 5)
SEEDS = [1, 2, 3, 4, 5]
NS3 = {'ns3.use_ns3': True, 'privacy.cpq_group_key': 'ml_kem'}
CPQ_LAMBDA = {'privacy.cpq_lambda': 0.75}
# attack: 10 origins, 5 vehicles each
ORIGINS_10 = [['E0'], ['E1'], ['E3'], ['E11'], ['E8'], ['-E1'], ['-E11'], ['-E8'], ['-E13'], ['E14']]

# experiment -> (output folder, what it reproduces)
EXPERIMENTS = {
    'E1': ('E1_privacy_utility', 'Fig. 2, Table 2, Figs. J.1, J.3, J.4'),
    'LAMBDA': ('cpq_lambda_ablation', 'Table 3, Fig. J.2'),
    'POPULATION': ('population_size', 'Table J.1'),
    'E2': ('E2_v2i_communication', 'Figs. 3, 4, Table 5'),
    'E3': ('E3_signature_schemes', 'Table 4, Figs. I.1, K.1'),
    'ATTACK': ('origin_attack', 'Fig. 5 (run attack/ on the logged data)'),
}


def run(method, seed, eps=None, **changes):
    """One run: method, seed, privacy epsilon (None for method none) and config changes
    given as section.key = value."""
    return {'method': method, 'seed': seed, 'eps': eps, 'changes': changes}


def matrix(experiment):
    if experiment == 'E1':
        return ([run('none', s) for s in SEEDS] +
                [run(m, s, e) for e in EPS for m in METHODS for s in SEEDS] +
                [run('cpq', s, e, **CPQ_LAMBDA) for e in EPS for s in SEEDS])
    if experiment == 'LAMBDA':
        seeds = range(1, 11)
        return ([run('cpq', s, 1, **{'privacy.cpq_lambda': lam}) for lam in (0.25, 0.5, 0.75, 0.9) for s in seeds] +
                [run(m, s, 1) for m in ('cpq', 'ldpfl') for s in seeds])
    if experiment == 'POPULATION':
        return [run(m, s, None if m == 'none' else 1, **{'sumo.num_clients': n})
                for n in (10, 25) for m in ('none', 'cpq', 'ldpfl') for s in SEEDS]
    if experiment == 'E2':
        return ([run(m, s, None if m == 'none' else 1, **NS3) for m in ['none'] + METHODS for s in SEEDS] +
                [run('cpq', s, 1, **NS3, **CPQ_LAMBDA) for s in SEEDS])
    if experiment == 'E3':
        return [run(m, 1, None if m == 'none' else 1, **NS3,
                    **{'ns3.pqc_scheme': scheme, 'experiment.true_rounds': 30})
                # liboqs names: SLH-DSA-SHA2-128f (pure) is SLH_DSA_PURE_SHA2_128F
                for scheme in ('ML-DSA-44', 'Falcon-512', 'SLH_DSA_PURE_SHA2_128F') for m in ('cpq', 'none')]
    if experiment == 'ATTACK':
        o = {**NS3, 'sumo.start_edges': ORIGINS_10}
        return ([run('none', s, **o) for s in (1, 2, 3)] +
                [run(m, s, e, **o) for e in (0.3, 1) for m in ('cpq', 'ldpfl') for s in (1, 2, 3)])
    raise SystemExit(f'unknown experiment {experiment}')


def run_name(r):
    """<method>[_lam<l>][_N<n>][_bg<period>][_<scheme>][_eps<eps>]_seed<s>"""
    c = r['changes']
    parts = [r['method']]
    if 'privacy.cpq_lambda' in c:
        parts.append(f"lam{c['privacy.cpq_lambda']:g}")
    if 'sumo.num_clients' in c:
        parts.append(f"N{c['sumo.num_clients']}")
    if 'sumo.background_period' in c:
        parts.append(f"bg{c['sumo.background_period']:g}")
    if 'ns3.pqc_scheme' in c:
        parts.append(c['ns3.pqc_scheme'])
    if r['eps'] is not None:
        parts.append(f"eps{r['eps']:g}")
    parts.append(f"seed{r['seed']}")
    return '_'.join(parts)


def run_config(base, r):
    cfg = copy.deepcopy(base)
    cfg['experiment']['method'] = r['method']
    if r['eps'] is not None:
        cfg['privacy']['epsilon'] = float(r['eps'])
    for key, value in r['changes'].items():
        section, name = key.split('.')
        if name not in cfg[section]:
            raise SystemExit(f'{key} is not in the base config')
        cfg[section][name] = value
    return cfg


def rounds_done(run_dir):
    path = os.path.join(run_dir, 'round_times.csv')
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def live_pid(run_dir, name):
    """pid of the process still working on this run (started by an earlier launcher), else None."""
    try:
        with open(os.path.join(run_dir, 'meta.json')) as f:
            pid = json.load(f)['pid']
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return pid if f'{name}.yaml'.encode() in f.read() else None
    except (OSError, KeyError, ValueError):
        return None


def git(*args):
    return subprocess.run(['git'] + list(args), capture_output=True, text=True, check=True).stdout.strip()


def available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable'):
                return int(line.split()[1]) / 1e6
    return float('inf')


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def main():
    p = argparse.ArgumentParser(description='\n'.join(f'{k:11s} {v[1]}' for k, v in EXPERIMENTS.items()),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment', choices=list(EXPERIMENTS))
    p.add_argument('--eps', type=float, nargs='*', help='only runs with these privacy epsilons (and the none runs)')
    p.add_argument('--methods', nargs='*', help='only these methods')
    p.add_argument('--seeds', type=int, nargs='*', help='only these seeds')
    p.add_argument('--max-parallel', type=int, default=3)
    p.add_argument('--stagger', type=float, default=45, help='s between two starts (shared startup file sumo/add/dqn.det.xml)')
    p.add_argument('--min-free-gb', type=float, default=3.0, help='wait with the next start while less memory is available')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    exp_dir = os.path.join(ROOT, EXPERIMENTS[args.experiment][0])
    with open(BASE) as f:
        base = yaml.safe_load(f)

    runs = matrix(args.experiment)
    if args.eps is not None:
        runs = [r for r in runs if r['eps'] is None or r['eps'] in args.eps]
    if args.methods:
        runs = [r for r in runs if r['method'] in args.methods]
    if args.seeds:
        runs = [r for r in runs if r['seed'] in args.seeds]

    todo, adopted = [], {}   # adopted: runs of an earlier launcher that are still going (name -> pid)
    for r in runs:
        name = run_name(r)
        cfg = run_config(base, r)
        run_dir = os.path.join(exp_dir, 'runs', name)
        done, total = rounds_done(run_dir), cfg['experiment']['true_rounds']
        pid = live_pid(run_dir, name)
        state = 'done' if done >= total else 'running' if pid else ('unfinished' if os.path.exists(run_dir) else 'new')
        print(f'{name:40s} {state:10s} {done}/{total} rounds')
        if state == 'running':
            adopted[name] = pid
        elif state != 'done':
            todo.append((name, cfg, r, run_dir))
    if args.dry_run or not (todo or adopted):
        return

    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty:
        raise SystemExit('uncommitted changes to tracked files; commit them first:\n' + dirty)
    commit = git('rev-parse', 'HEAD')

    for sub in ('configs', 'runs', 'results'):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    os.makedirs(ROOT, exist_ok=True)
    frozen = os.path.join(ROOT, 'base_config.yaml')
    if not os.path.exists(frozen):
        shutil.copyfile(BASE, frozen)

    env_base = dict(os.environ, CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3', PYTHONUNBUFFERED='1')
    prefix = os.environ.get('CONDA_PREFIX')
    if prefix:
        env_base['LD_LIBRARY_PATH'] = f"{prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    running = {}   # name -> (process, log file, meta, meta path)

    def collect():
        for name, pid in list(adopted.items()):
            if not os.path.exists(f'/proc/{pid}'):
                print(f'[{now()}] finished {name} (started by an earlier launcher)', flush=True)
                del adopted[name]
        for name in list(running):
            proc, log, meta, meta_path = running[name]
            if proc.poll() is None:
                continue
            log.close()
            meta.update(end=now(), exit_code=proc.returncode, wall_s=round(time.time() - meta['t0'], 1))
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=1)
            print(f'[{now()}] finished {name} exit={proc.returncode} wall={meta["wall_s"] / 3600:.2f} h', flush=True)
            del running[name]

    last_start = 0.0
    for name, cfg, r, run_dir in todo:
        while True:
            collect()
            busy = len(running) + len(adopted)
            if (busy < args.max_parallel and time.time() - last_start >= args.stagger
                    and (not busy or available_gb() >= args.min_free_gb)):
                break
            time.sleep(5)
        if os.path.exists(run_dir):
            failed = os.path.join(exp_dir, 'runs', '_failed')
            os.makedirs(failed, exist_ok=True)
            shutil.move(run_dir, os.path.join(failed, f"{name}_{datetime.datetime.now():%Y%m%d-%H%M%S}"))
        os.makedirs(run_dir)
        cfg_path = os.path.join(exp_dir, 'configs', name + '.yaml')
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        fl_env = {'FL_OUT_DIR': run_dir, 'FL_SEED': str(r['seed']), 'FL_RUN_ID': f'{args.experiment}_{name}'}
        if args.experiment == 'ATTACK':
            fl_env['FL_ATTACK_LOG'] = '1'
        if args.experiment == 'LAMBDA':
            fl_env['FL_MSE_LOG'] = '1'
        cmd = [sys.executable, 'run_shared.py', '--config', cfg_path]
        log = open(os.path.join(run_dir, 'run.log'), 'w')
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=dict(env_base, **fl_env), start_new_session=True)
        meta = {'experiment': args.experiment, 'run': name, 'commit': commit, 'host': socket.gethostname(),
                'command': ' '.join(cmd), 'env': fl_env, 'changes': dict(r['changes'], method=r['method'], eps=r['eps']),
                'pid': proc.pid, 'start': now(), 't0': time.time()}
        meta_path = os.path.join(run_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=1)
        running[name] = (proc, log, meta, meta_path)
        last_start = time.time()
        print(f'[{now()}] started {name} pid={proc.pid}', flush=True)

    while running or adopted:
        collect()
        time.sleep(5)


if __name__ == '__main__':
    main()
