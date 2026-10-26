"""Result tables of one experiment from its run folders.

    python tools/analyze_experiment.py experiments/final/E1_privacy_utility [--window 10]

Writes <experiment>/results/: summary.csv (one row per run, metrics averaged over the last 10 rounds),
grouped.csv (mean and 95% CI over seeds per setting) and curves_<metric>.csv (one value per round).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

# two-sided 95% t quantile by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
SETTINGS = ['method', 'eps', 'cpq_lambda', 'num_clients', 'background_period', 'ns3', 'pqc_scheme', 'rounds']
CURVES = ['travel_time_mean', 'arrival_rate', 'time_loss_mean']


def read(run_dir, name):
    path = os.path.join(run_dir, name)
    return pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else None


def first_below(series, level, width=5):
    """First round whose `width`-round mean (ending at that round) is below `level`."""
    rolling = series.rolling(width).mean()
    hit = rolling[rolling < level]
    return int(hit.index[0]) if len(hit) else np.nan


def summarize(run_dir, window):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    meta = {}
    if os.path.exists(os.path.join(run_dir, 'meta.json')):
        with open(os.path.join(run_dir, 'meta.json')) as f:
            meta = json.load(f)
    method, rounds = cfg['experiment']['method'], cfg['experiment']['true_rounds']
    ns3 = bool(cfg['ns3']['use_ns3'])
    row = {
        'run': os.path.basename(run_dir), 'method': method,
        'eps': np.nan if method == 'none' else cfg['privacy']['epsilon'],
        'cpq_lambda': cfg['privacy']['cpq_lambda'] if method == 'cpq' else np.nan,
        'num_clients': cfg['sumo']['num_clients'], 'background_period': cfg['sumo']['background_period'],
        'ns3': ns3, 'pqc_scheme': cfg['ns3']['pqc_scheme'] if ns3 else '', 'rounds': rounds,
        'seed': int(meta.get('env', {}).get('FL_SEED', cfg['experiment']['seeds'][0])),
        'commit': meta.get('commit', '')[:7], 'exit_code': meta.get('exit_code', np.nan),
        'wall_h': round(meta['wall_s'] / 3600, 2) if 'wall_s' in meta else np.nan,
    }
    ev = read(run_dir, 'global_eval.csv')
    ev = ev.set_index('round') if ev is not None else pd.DataFrame()
    row['rounds_done'] = len(ev)
    problems = []
    if len(ev) < rounds:
        problems.append(f'{len(ev)}/{rounds} rounds')
        return row, ev, problems
    last = ev.loc[rounds - window + 1:rounds]
    row.update(
        travel_final=last['travel_time_mean'].mean(), travel_final_std_rounds=last['travel_time_mean'].std(),
        arrival_final=last['arrival_rate'].mean(), time_loss_final=last['time_loss_mean'].mean(),
        waiting_final=last['waiting_time_mean'].mean(),
        conv_round_100=first_below(ev['travel_time_mean'], 100), conv_round_150=first_below(ev['travel_time_mean'], 150),
        best_round=int(ev['travel_time_mean'].idxmin()), best_travel=ev['travel_time_mean'].min())
    base = read(run_dir, 'baselines.csv')
    if base is not None:
        base = base.set_index('round')
        row['shortest_travel'], row['reroute_travel'] = base.loc['shortest', 'travel_time_mean'], base.loc['reroute', 'travel_time_mean']
    weights = read(run_dir, 'global_weights.csv')
    if weights is not None:
        row['max_abs_weight'] = weights['max_abs'].max()
        if not weights['finite'].all() or not np.isfinite(weights['max_abs']).all():
            problems.append('non-finite weights')
    times = read(run_dir, 'round_times.csv')
    if times is not None:
        row['round_s'] = times['round_s'].mean()
    part, comm = read(run_dir, 'participation.csv'), read(run_dir, 'comm.csv')
    if ns3 and part is not None and comm is not None:
        row['aggregated_per_round'] = part['aggregated'].mean()
        row['missing_share'] = 1 - part['aggregated'].sum() / (row['num_clients'] * len(part))
        row['downlink_delivered_share'] = part['downlink_delivered'].sum() / part['downlink_sent'].sum()
        row['uplink_delivered_share'] = part['uplink_delivered'].sum() / max(1, part['uplink_sent'].sum())
        model = comm[comm['Direction'].isin(['downlink', 'uplink'])]
        for direction in ('downlink', 'uplink'):
            sent = model[model['Direction'] == direction]
            ok = sent[sent['Outcome'] == 'delivered']
            row[f'{direction}_attempt_success'] = len(ok) / max(1, len(sent))
            row[f'{direction}_latency_median'] = ok['Duration'].median()
            row[f'{direction}_latency_p90'] = ok['Duration'].quantile(0.9)
            row[f'{direction}_wire_kb'] = ok['Wire_Bytes'].median() / 1000
            row[f'{direction}_kb_per_round'] = sent['Wire_Bytes'].sum() / 1000 / len(part)
        row['rsu_busy_s_per_round'] = model['Duration'].sum() / len(part)
        setup = comm[comm['Direction'] == 'setup']
        if len(setup):
            row['setup_kb'] = setup['Wire_Bytes'].sum() / 1000
    log = os.path.join(run_dir, 'run.log')
    if os.path.exists(log):
        with open(log, errors='replace') as f:
            text = f.read()
        if 'Traceback' in text:
            problems.append('Traceback in run.log')
        if 'nan' in text.lower().replace('finance', ''):
            problems.append('"nan" in run.log')
    if row['exit_code'] not in (0, 0.0) and not pd.isna(row['exit_code']):
        problems.append(f"exit code {row['exit_code']}")
    return row, ev, problems


def mean_ci(values):
    values = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return values[0], np.nan
    return values.mean(), T95.get(n - 1, 1.96) * values.std(ddof=1) / np.sqrt(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_dir')
    p.add_argument('--window', type=int, default=10, help='final rounds averaged (default: last 10)')
    args = p.parse_args()
    runs_dir = os.path.join(args.experiment_dir, 'runs')
    out_dir = os.path.join(args.experiment_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)

    rows, curves, problems = [], {}, []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if name.startswith('_') or not os.path.exists(os.path.join(run_dir, 'config.yaml')):
            continue
        row, ev, bad = summarize(run_dir, args.window)
        problems += [f'{name}: {b}' for b in bad]
        if 'travel_final' in row:
            rows.append(row)
            curves[name] = ev
    if not rows:
        print('no finished runs;', '; '.join(problems))
        return
    summary = pd.DataFrame(rows)

    # difference to the no-privacy run with the same seed and the same world
    world = ['num_clients', 'background_period', 'ns3', 'pqc_scheme', 'rounds', 'seed']
    ref = summary[summary['method'] == 'none'].set_index(world)[['travel_final', 'arrival_final']]
    for col in ('travel_final', 'arrival_final'):
        keys = [tuple(k) for k in summary[world].values]
        summary[f'{col}_minus_none'] = [summary.loc[i, col] - ref.loc[k, col] if k in ref.index else np.nan
                                        for i, k in zip(summary.index, keys)]
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False, float_format='%.4g')

    metrics = [c for c in summary.columns if c not in SETTINGS + ['run', 'seed', 'commit', 'exit_code', 'rounds_done']]
    grouped = []
    for key, g in summary.groupby(SETTINGS, dropna=False, sort=False):
        out = dict(zip(SETTINGS, key), seeds=len(g))
        for m in metrics:
            out[m], out[m + '_ci95'] = mean_ci(g[m])
        grouped.append(out)
    grouped = pd.DataFrame(grouped).dropna(axis=1, how='all')
    grouped.to_csv(os.path.join(out_dir, 'grouped.csv'), index=False, float_format='%.4g')
    for metric in CURVES:
        pd.DataFrame({name: ev[metric] for name, ev in curves.items()}).to_csv(
            os.path.join(out_dir, f'curves_{metric}.csv'), float_format='%.4g')

    # baselines must be identical for runs that share traffic density, vehicles and seed-independent scenarios
    for key, g in summary.groupby(['num_clients', 'background_period']):
        if g['shortest_travel'].nunique() > 1 or g['reroute_travel'].nunique() > 1:
            problems.append(f'baselines differ between runs with N={key[0]}, traffic {key[1]} s')

    show = ['method', 'eps', 'seeds', 'travel_final', 'travel_final_ci95', 'arrival_final', 'arrival_final_ci95',
            'travel_final_minus_none', 'travel_final_minus_none_ci95', 'conv_round_100', 'max_abs_weight']
    extra = [c for c in ('cpq_lambda', 'num_clients', 'background_period', 'pqc_scheme') if c in grouped and grouped[c].nunique(dropna=False) > 1]
    extra += [c for c in ('aggregated_per_round', 'uplink_delivered_share') if c in grouped]
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.3g}'.format):
        print(grouped[[c for c in show[:2] + extra + show[2:] if c in grouped]].to_string(index=False))
    print(f'\n{len(summary)} finished runs -> {out_dir}/summary.csv, grouped.csv, curves_*.csv')
    print('checks:', 'all passed' if not problems else '')
    for line in problems:
        print('  PROBLEM', line)


if __name__ == '__main__':
    main()
