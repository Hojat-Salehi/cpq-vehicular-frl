"""Tables and figure data of the paper from the run folders in experiments/final/ (own runs or the
Zenodo archive).

    python tools/paper_results.py                    # writes results/paper/tables.md and figure_data.json
    bash figures/make_all.sh --data results/paper/figure_data.json   # figures from these data

Experiments that are missing are skipped.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_experiment import T95, summarize  # noqa: E402

ROOT = os.path.join('experiments', 'final')
EPS = (0.3, 0.5, 1.0, 2.0, 5.0)
LABEL = {'none': 'No privacy', 'cpq': 'CPQ', 'cpql': 'CPQ-λ', 'ldpfl': 'LDP-FL',
         'gaussian': 'Gaussian', 'laplace': 'Laplace'}
ORDER = ['cpq', 'cpql', 'ldpfl', 'gaussian', 'laplace']
SCHEMES = {'ML-DSA-44': 'ML-DSA-44', 'Falcon-512': 'Falcon-512', 'SLH_DSA_PURE_SHA2_128F': 'SLH-DSA-SHA2-128f'}


def t95(n):
    return T95[n - 1]


def load(folder):
    """One row per finished run (tools/analyze_experiment.summarize) plus the run's evaluation per round."""
    rows, curves = [], {}
    for run_dir in sorted(glob.glob(os.path.join(ROOT, folder, 'runs', '*'))):
        if os.path.basename(run_dir).startswith('_') or not os.path.exists(os.path.join(run_dir, 'config.yaml')):
            continue
        row, ev, problems = summarize(run_dir, 10)
        if problems:
            raise SystemExit(f'{run_dir}: {problems}')
        lam = row['cpq_lambda']
        row['key'] = 'cpql' if row['method'] == 'cpq' and lam == 0.75 else row['method']
        row['lam'] = lam
        rows.append(row)
        curves[row['run']] = ev
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    # run-level values with 4 significant digits, as in the summary.csv the paper's tables were computed from
    num = df.select_dtypes('number').columns.drop(['seed', 'eps', 'lam', 'num_clients'], errors='ignore')
    df[num] = df[num].apply(lambda c: c.map(lambda x: float(f'{x:.4g}') if pd.notna(x) else x))
    return df, curves


def blocks(curves, runs, col, scale=1.0):
    """Mean over the runs of each 10-round block."""
    m = pd.concat([curves[r][col] for r in runs], axis=1).mean(axis=1) * scale
    return [round(float(m.loc[a:a + 9].mean()), 1) for a in range(1, 101, 10)]


def ms(v, d=1):
    v = np.asarray(v, float)
    return f'{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}'


def mci(v, d=1):
    v = np.asarray(v, float)
    return f'{v.mean():.{d}f} ± {t95(len(v)) * v.std(ddof=1) / math.sqrt(len(v)):.{d}f}'


def sel(df, key, eps=None, **kw):
    g = df[df.key == key]
    if eps is not None:
        g = g[np.isclose(g.eps, eps)]
    for k, v in kw.items():
        g = g[g[k] == v]
    return g.sort_values('seed')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=os.path.join('results', 'paper'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md, data = [], {}
    w = md.append

    e1, c1 = load('E1_privacy_utility')
    if e1 is not None:
        none = sel(e1, 'none')
        w('## Table 2: privacy-utility under ideal communication (mean ± SD over seeds)\n')
        for col, title, sc in [('travel_final', 'Travel time (s)', 1), ('arrival_final', 'Arrival rate (%)', 100)]:
            w(f'{title}\n\n| Mechanism | ' + ' | '.join(f'ε = {e:g}' for e in EPS) + ' |\n|---|' + '---|' * len(EPS))
            for k in ORDER:
                w(f'| {LABEL[k]} | ' + ' | '.join(ms(sel(e1, k, e)[col] * sc) for e in EPS) + ' |')
            w(f'| No privacy | {ms(none[col] * sc)} | | | | |\n')
        travel = {'No privacy': blocks(c1, none.run, 'travel_time_mean')}
        for k in ORDER:
            travel[LABEL[k]] = {str(e): blocks(c1, sel(e1, k, e).run, 'travel_time_mean') for e in EPS}
        data['fig2'] = travel
        per_seed, reported = {}, {}
        for metric, col, sc in [('travel', 'travel_final', 1), ('arrival', 'arrival_final', 100),
                                ('loss', 'time_loss_final', 1), ('waiting', 'waiting_final', 1)]:
            per_seed[metric] = {'No privacy|none': [round(x * sc, 1) for x in none[col]]}
            reported[metric] = {'No privacy|none': [round(none[col].mean() * sc, 1), round(none[col].std() * sc, 1)]}
            for k in ('cpq', 'ldpfl', 'gaussian', 'laplace'):
                for e in EPS:
                    g = sel(e1, k, e)[col] * sc
                    per_seed[metric][f'{LABEL[k]}|{e:g}'] = [round(x, 1) for x in g]
                    reported[metric][f'{LABEL[k]}|{e:g}'] = [round(g.mean(), 1), round(g.std(), 1)]
        data['appendix_e1'] = {'DATA': per_seed, 'REPORTED': reported}

    la, cl = load('cpq_lambda_ablation')
    if la is not None:
        w('## Table 3: CPQ-λ ablation at ε = 1 (MSE mean ± 95% CI over the runs with mse.csv; utility over all seeds)\n')
        w('| λ | MSE (×10⁻⁵) | Travel time (s) | Arrival (%) |\n|---|---|---|---|')
        groups = [('LDP-FL', la[la.method == 'ldpfl'])] + \
                 [(f'{l:g}', la[(la.method == 'cpq') & np.isclose(la.lam, l)]) for l in (0.25, 0.5, 0.75, 0.9)] + \
                 [('1', la[(la.method == 'cpq') & np.isclose(la.lam, 1.0)])]
        travel10 = {}
        for name, g in groups:
            g = g.sort_values('seed')
            mse = []
            for r in g.run:
                f = os.path.join(ROOT, 'cpq_lambda_ablation', 'runs', r, 'mse.csv')
                if os.path.exists(f):
                    mse.append(pd.read_csv(f)['mse'].mean() * 1e5)
            w(f'| {name} | {mci(mse, 3) if len(mse) > 1 else "–"} | {mci(g.travel_final, 0)} | {mci(g.arrival_final * 100, 0)} |')
            # whole seconds, as in Fig. J.2
            travel10[name] = [round(cl[r]['travel_time_mean'].iloc[-10:].mean()) for r in g.run]
        w('')
        data['appendix_lambda'] = {'LDP-FL': travel10['LDP-FL'], **{k: v for k, v in travel10.items() if k != 'LDP-FL'}}

    po, _ = load('population_size')
    if po is not None:
        w('## Table J.1: number of learning vehicles at ε = 1 (paired over seeds, mean ± 95% CI)\n')
        w('| K | Travel-time gain LDP-FL − CPQ (s) | Arrival gain CPQ − LDP-FL (pp) | Seeds where CPQ is faster |\n|---|---|---|---|')
        sets = [(n, po[po.num_clients == n]) for n in (10, 25)]
        if e1 is not None:
            sets.append((50, e1[np.isclose(e1.eps.fillna(-1), 1.0) | (e1.key == 'none')]))
        for n, g in sets:
            c = sel(g, 'cpq', 1.0).set_index('seed')
            l = sel(g, 'ldpfl', 1.0).set_index('seed')
            d_t, d_a = l.travel_final - c.travel_final, (c.arrival_final - l.arrival_final) * 100
            w(f'| {n} | {mci(d_t)} | {mci(d_a)} | {int((d_t > 0).sum())}/{len(d_t)} |')
        w('')

    e2, c2 = load('E2_v2i_communication')
    if e2 is not None:
        w('## Table 5: 802.11p with ML-DSA-44 at ε = 1 (mean ± SD over seeds)\n')
        w('| Method | Travel (s) | Arrival (%) | DL reach (%) | UL delivery (%) | Updates per round | ΔT vs ideal (s) |\n'
          '|---|---|---|---|---|---|---|')
        for k in ['none'] + ORDER:
            g = sel(e2, k).set_index('seed')
            dt = '–'
            if e1 is not None:
                ref = sel(e1, k, None if k == 'none' else 1.0).set_index('seed').travel_final
                dt = ms(g.travel_final - ref, 1)
            w(f'| {LABEL[k]} | {ms(g.travel_final)} | {ms(g.arrival_final * 100)} | {ms(g.downlink_delivered_share * 100)} | '
              f'{ms(g.uplink_delivered_share * 100)} | {ms(g.aggregated_per_round)} | {dt} |')
        w('')
        data['fig4'] = {
            'travel': {LABEL[k]: blocks(c2, sel(e2, k).run, 'travel_time_mean') for k in ['none'] + ORDER},
            'aggregated': {},
        }
        for k in ['none'] + ORDER:
            parts = [pd.read_csv(os.path.join(ROOT, 'E2_v2i_communication', 'runs', r, 'participation.csv'))
                     .set_index('round')['aggregated'] for r in sel(e2, k).run]
            m = pd.concat(parts, axis=1).mean(axis=1)
            data['fig4']['aggregated'][LABEL[k]] = [round(float(m.loc[a:a + 9].mean()), 1) for a in range(1, 101, 10)]
        f3 = {'TRAFFIC_KB': {}, 'UPLINK_LATENCY_S': {}, 'UPLINK_SUCCESS': {}}
        bins = [0, 50, 75, 100, 125, 150, 175, 200]
        for k in ('none', 'cpq', 'ldpfl', 'gaussian', 'laplace'):
            g = sel(e2, k)
            f3['TRAFFIC_KB'][LABEL[k]] = [round(g.downlink_kb_per_round.mean(), 1), round(g.uplink_kb_per_round.mean(), 1)]
            f3['UPLINK_LATENCY_S'][LABEL[k]] = [round(g.uplink_latency_median.mean(), 3), round(g.uplink_latency_p90.mean(), 3)]
            comm = pd.concat([pd.read_csv(os.path.join(ROOT, 'E2_v2i_communication', 'runs', r, 'comm.csv')) for r in g.run])
            up = comm[comm.Direction == 'uplink']
            b = pd.cut(up.Dist_Start, bins, right=False)
            n = up.groupby(b, observed=False).size()
            ok = up[up.Outcome == 'delivered'].groupby(b[up.Outcome == 'delivered'], observed=False).size()
            f3['UPLINK_SUCCESS'][LABEL[k]] = [[None if x == 0 else round(100 * y / x, 1) for x, y in zip(n, ok)],
                                              [int(x) for x in n]]
        data['fig3'] = f3

    e3, _ = load('E3_signature_schemes')
    if e3 is not None:
        w('## Table 4: signature schemes, uplink (one seed, 30 rounds)\n')
        w('| Scheme | Update | UL size (KB) | Attempt success (%) | UL delivery (%) |\n|---|---|---|---|---|')
        succ, setup = {}, {}
        for sc, name in SCHEMES.items():
            for k, upd in (('none', 'FP'), ('cpq', 'CPQ')):
                r = e3[(e3.key == k) & (e3.pqc_scheme == sc)].iloc[0]
                w(f'| {name} | {upd} | {r.uplink_wire_kb:.2f} | {100 * r.uplink_attempt_success:.1f} | {100 * r.uplink_delivered_share:.1f} |')
                comm = pd.read_csv(os.path.join(ROOT, 'E3_signature_schemes', 'runs', r.run, 'comm.csv'))
                up = comm[comm.Direction == 'uplink']
                pct = []
                for lo in (75, 100, 125, 150):
                    x = up[(up.Dist_Start >= lo) & (up.Dist_Start < lo + 25)]
                    pct.append(round(100 * (x.Outcome == 'delivered').mean()))
                succ[f'{name}|{"No privacy" if k == "none" else "CPQ"}'] = pct
                log = os.path.join(ROOT, 'E3_signature_schemes', 'runs', r.run, 'setup.log')
                if k == 'cpq' and os.path.exists(log):
                    setup[name] = [int(x) for x in re.findall(r'SETUP_OK verified (\d+)/', open(log).read())]
        w('')
        data['appendix_e3'] = {'UPLINK_SUCCESS': succ, 'SETUP_OK': setup}

    att = os.path.join(ROOT, 'origin_attack', 'results', 'attack')
    if os.path.exists(os.path.join(att, 'fig_rounds_data.csv')):
        fig5 = {}
        for scen, f, extra in (('shadow', 'fig_rounds_data.csv', None), ('known', 'fig_known_data.csv', 0.2)):
            d = pd.read_csv(os.path.join(att, f))
            d = d[d.method == 'headline']
            if extra is not None:
                d = d[np.isclose(d.known_frac, extra)]
            fig5[scen] = {}
            for mech, eps in (('none', np.inf), ('cpq', 0.3), ('ldpfl', 0.3)):
                g = d[(d.mechanism == mech) & np.isclose(d.eps, eps)].sort_values('k')
                fig5[scen][LABEL[mech]] = [[round(100 * x) for x in g[c]] for c in ('acc_mean', 'ci_low', 'ci_high')]
        data['fig5'] = fig5
        w('Figure 5 data: attack accuracy (headline attack) from results/attack/fig_rounds_data.csv.\n')

    with open(os.path.join(args.out, 'tables.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(args.out, 'figure_data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('\n'.join(md))
    print(f'-> {args.out}/tables.md, {args.out}/figure_data.json')


if __name__ == '__main__':
    main()
