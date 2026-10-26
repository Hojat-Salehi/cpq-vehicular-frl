"""Launch the runs of one experiment.

    python tools/run_experiment.py E1 --dry-run   # list the runs and their state
    python tools/run_experiment.py E1             # run them (3 in parallel)
    python tools/run_experiment.py E1 --eps 1     # only the eps = 1 runs

Each run is configs/base.yaml plus its changes; configs, run outputs and logs go to
experiments/final/<experiment>/. Finished runs are skipped, so the command can be repeated.
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
SEEDS = [1, 2, 3, 4, 5]
NS3 = {'ns3.use_ns3': True, 'privacy.cpq_group_key': 'ml_kem'}

FOLDERS = {
    'P1': 'pilots/P1_traffic_1.5', 'P2': 'pilots/P2_setup_50',
    'E1': 'E1_privacy_utility', 'E2': 'E2_ns3_communication', 'E3': 'E3_pqc_schemes',
    'A1': 'A1_cpq_lambda', 'A2': 'A2_num_vehicles', 'A3': 'A3_traffic_density',
    'E1X': 'E1X_eps3_seeds6-10', 'ATK': 'ATK_origin_inference', 'ATKN': 'ATK_ns3_10origins',
    'A1X': 'A1X_cpq_lambda_mse', 'E2L': 'E2L_ns3_cpq_lambda0.75', 'E1L': 'E1L_cpq_lambda0.75',
    'E1S': 'E1S_seeds6-10',
}
# 2026-10-19 extension: E1 with eps 0.3, 0.5, 1, 2, 5 (eps 3 dropped), 10 seeds, CPQ-lambda 0.75 as a method
E1_EPS = (0.3, 0.5, 1, 2, 5)
NEW_SEEDS = [6, 7, 8, 9, 10]
# ATKN: 10 origins (5 vehicles each); the 5 added ones pass the RSU at a distance like the base 5 (120-143 m on
# the shortest route), so upload success alone does not reveal the origin
ORIGINS_10 = [['E0'], ['E1'], ['E3'], ['E11'], ['E8'], ['-E1'], ['-E11'], ['-E8'], ['-E13'], ['E14']]
ATK_SEEDS = [1, 2, 3, 4]


def run(method, seed, eps=None, **changes):
    """One run: method, seed, privacy epsilon (None for method none) and config changes
    given as section.key = value."""
    return {'method': method, 'seed': seed, 'eps': eps, 'changes': changes}


def matrix(experiment):
    if experiment == 'P1':
        return [run('none', 1, **{'sumo.background_period': 1.5}),
                run('none', 1, **{'sumo.background_period': 1.75}),
                run('cpq', 1, 1, **{'sumo.background_period': 1.5})]
    if experiment == 'P2':
        return [run('cpq', 1, 1, **NS3, **{'experiment.true_rounds': 3})]
    if experiment == 'E1':
        # order: eps 1, 3, then the one-bit methods at eps 5 (same noise level as eps 3: a second look
        # at CPQ vs LDP-FL there), then 0.5, 0.3, 2 (added 2026-10-11), then the additive methods at eps 5
        blocks = [(1, METHODS), (3, METHODS), (5, ['cpq', 'ldpfl']), (0.5, METHODS), (0.3, METHODS),
                  (2, METHODS), (5, ['gaussian', 'laplace'])]
        return ([run('none', s) for s in SEEDS] +
                [run(m, s, e) for e, methods in blocks for m in methods for s in SEEDS])
    if experiment == 'E2':
        return [run(m, s, None if m == 'none' else 1, **NS3)
                for m in ['none'] + METHODS for s in SEEDS]
    if experiment == 'E3':
        return [run(m, 1, None if m == 'none' else 1, **NS3,
                    **{'ns3.pqc_scheme': scheme, 'experiment.true_rounds': 30})
                # liboqs names: SLH-DSA-SHA2-128f (pure) is SLH_DSA_PURE_SHA2_128F
                for scheme in ('ML-DSA-44', 'Falcon-512', 'SLH_DSA_PURE_SHA2_128F') for m in ('cpq', 'none')]
    if experiment == 'E1X':
        # appendix: eps 3 of E1 with seeds 6-10, so that eps 3 has 10 seeds together with E1
        return [run('none', s) for s in range(6, 11)] + [run(m, s, 3) for m in METHODS for s in range(6, 11)]
    if experiment == 'ATK':
        # origin inference attack (attack/, Attack_Exp/): E1 settings, the server's observations logged
        # (FL_ATTACK_LOG=1); pilot: --methods none cpq ldpfl --eps 0.3 5 --seeds 1 2
        return ([run('none', s) for s in ATK_SEEDS] +
                [run(m, s, e) for e in (0.3, 1, 5) for m in METHODS for s in ATK_SEEDS])
    if experiment == 'ATKN':
        # origin inference attack, realistic: ns-3 (as E2), 10 origins, server observations logged (FL_ATTACK_LOG=1)
        o = {**NS3, 'sumo.start_edges': ORIGINS_10}
        return ([run('none', s, **o) for s in (1, 2, 3)] +
                [run(m, s, e, **o) for e in (0.3, 1) for m in ('cpq', 'ldpfl') for s in (1, 2, 3)])
    if experiment == 'A1':
        return [run('cpq', s, 1, **{'privacy.cpq_lambda': lam})
                for lam in (0.25, 0.5, 0.75, 0.9) for s in SEEDS]
    if experiment == 'A1X':
        # lambda ablation at eps 1 with the aggregation MSE (FL_MSE_LOG=1): lambda 0 (LDP-FL through the
        # CPQ-lambda code) to 0.9 on seeds 6-10, lambda 0 also on seeds 1-5; CPQ (lambda 1) and LDP-FL on
        # seeds 6-10 (these two are also E1's eps 1 runs for seeds 6-10)
        lam = lambda l, s: run('cpq', s, 1, **{'privacy.cpq_lambda': l})
        return ([lam(l, s) for l in (0.0, 0.25, 0.5, 0.75, 0.9) for s in NEW_SEEDS] +
                [run(m, s, 1) for m in ('cpq', 'ldpfl') for s in NEW_SEEDS] +
                [lam(0.0, s) for s in SEEDS])
    if experiment == 'E2L':
        return [run('cpq', s, 1, **NS3, **{'privacy.cpq_lambda': 0.75}) for s in SEEDS]
    if experiment == 'E1L':
        # CPQ-lambda 0.75 in E1; eps 1 is in A1 (seeds 1-5) and A1X (seeds 6-10)
        return [run('cpq', s, e, **{'privacy.cpq_lambda': 0.75}) for e in E1_EPS if e != 1 for s in SEEDS + NEW_SEEDS]
    if experiment == 'E1S':
        # E1 seeds 6-10; none seeds 6-8 are in E1X, cpq / ldpfl eps 1 in A1X
        return ([run('none', s) for s in (9, 10)] +
                [run(m, s, e) for e in E1_EPS for m in METHODS for s in NEW_SEEDS
                 if not (e == 1 and m in ('cpq', 'ldpfl'))])
    if experiment == 'A2':
        return [run(m, s, None if m == 'none' else 1, **{'sumo.num_clients': n})
                for n in (10, 25) for m in ('none', 'cpq', 'ldpfl') for s in SEEDS]
    if experiment == 'A3':
        return [run(m, 1, None if m == 'none' else 1, **{'sumo.background_period': p})
                for p in (3, 2) for m in ('none', 'cpq', 'ldpfl')]
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
    p = argparse.ArgumentParser()
    p.add_argument('experiment', choices=sorted(FOLDERS))
    p.add_argument('--eps', type=float, nargs='*', help='only runs with these privacy epsilons (and the none runs)')
    p.add_argument('--methods', nargs='*', help='only these methods')
    p.add_argument('--seeds', type=int, nargs='*', help='only these seeds')
    p.add_argument('--max-parallel', type=int, default=3)
    p.add_argument('--stagger', type=float, default=45, help='s between two starts (shared startup file sumo/add/dqn.det.xml)')
    p.add_argument('--min-free-gb', type=float, default=3.0, help='wait with the next start while less memory is available')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    exp_dir = os.path.join(ROOT, FOLDERS[args.experiment])
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
        if args.experiment in ('ATK', 'ATKN'):
            fl_env['FL_ATTACK_LOG'] = '1'
        if args.experiment == 'A1X':
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
