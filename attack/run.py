"""Origin inference attack, single entry point (run from the repo root in conda env attack_env):

    python -m attack.run --stage {features,test,attack,aggregate,plot,all} --config attack/config.yaml

The "log" step of the spec is the simulation itself: tools/run_experiment.py ATK (FL_ATTACK_LOG=1).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from attack import dataset, features, infer, known


def configs(cfg):
    """{(method, eps): [cache dirs]} of finished runs, features built."""
    out = {}
    for (m, e), runs in sorted(dataset.finished_runs(cfg['exp_dir']).items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        out[(m, e)] = [features.build(r, os.path.join(cfg['out_dir'], 'cache'), cfg['pos_inputs']) for r in runs]
    return out


def stage_features(cfg):
    for (m, e), dirs in configs(cfg).items():
        print(f'features {m} eps={e}: {len(dirs)} runs', flush=True)


def stage_test(cfg):
    """Reconstruction test: decoding / reading the logged uploads with attack.dataset and averaging them
    gives the server's next global model (logged independently) in every round (with ns-3: the updates that
    arrived; a round without any keeps the global model); for one-bit methods every decoded coordinate is
    w_g +- alpha * r."""
    import onebit
    res = []
    for (m, e), runs in sorted(dataset.finished_runs(cfg['exp_dir']).items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        for r in runs:
            meta = dataset.load_meta(r)
            worst, worst_level = 0.0, 0.0
            for t in range(1, meta['rounds']):
                obs = dataset.load_round(r, t, meta)
                g = dataset.load_global(r, t)
                avg = np.mean(np.stack(list(obs.values())).astype(np.float64), 0) if obs else g
                worst = max(worst, float(np.abs(avg - dataset.load_global(r, t + 1)).max()))
                if m in onebit.METHODS and obs:
                    a = onebit.alpha(e) * 0.02     # clip radius 0.02 in every layer (checked below)
                    dev = np.abs(np.stack(list(obs.values())) - g)
                    worst_level = max(worst_level, float(np.abs(dev - a).max()))
            if m in onebit.METHODS:
                assert set(meta['clip_radius'].values()) == {0.02}, meta['clip_radius']
            ok = worst < 1e-5 and worst_level < 1e-5
            res.append({'run': r, 'method': m, 'eps': e, 'max_abs_diff_to_next_global': worst,
                        'max_abs_dev_from_alpha_r': worst_level, 'pass': ok})
            print(f"test {os.path.basename(r)}: next global diff {worst:.2e}, alpha*r level {worst_level:.2e} "
                  f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    os.makedirs(cfg['out_dir'], exist_ok=True)
    with open(os.path.join(cfg['out_dir'], 'reconstruction_test.json'), 'w') as f:
        json.dump(res, f, indent=1)
    if not all(x['pass'] for x in res):
        raise SystemExit('reconstruction test FAILED')


def _set(run, X, y=None):
    seed = run['info']['seed']
    return {'X': X, 'y': run['labels'] if y is None else y,
            'gid': np.array([f"{seed}:{v}" for v in run['info']['vehs']])}


def stage_attack(cfg):
    rng = np.random.default_rng(cfg['seed'])
    ks = cfg['ks']
    correct_rows, prob_frames, sel_rows = [], [], []
    for (m, e), dirs in configs(cfg).items():
        runs = [features.load(d) for d in dirs]
        if len(runs) < 2:
            print(f'skip {m} eps={e}: {len(runs)} run(s)', flush=True)
            continue
        K = len(np.unique(np.concatenate([r['labels'] for r in runs])))
        for i, test in enumerate(runs):
            t0 = time.time()
            train = [r for j, r in enumerate(runs) if j != i]
            test_seed = test['info']['seed']
            assert test_seed not in {r['info']['seed'] for r in train}, 'run seed in train and test'
            assert test['info']['run_dir'] not in {r['info']['run_dir'] for r in train}
            feats = {'pos': {r['info']['run_dir']: r['pos'] for r in runs},
                     'full': infer.pca_project(train, runs, cfg, rng),
                     'probe': {r['info']['run_dir']: r['probe'] for r in runs}}
            tr_sets = {fs: [_set(r, feats[fs][r['info']['run_dir']]) for r in train] for fs in feats}
            te_set = {fs: _set(test, feats[fs][test['info']['run_dir']]) for fs in feats}
            C = {fs: infer.choose_C(*infer._flatten(tr_sets[fs]), cfg) for fs in feats}

            # selection of feature set and classifier: per-round accuracy on vehicle-grouped inner folds
            inner = {fs: infer.inner_splits(tr_sets[fs], cfg) for fs in feats}
            score = {}
            for fs in feats:
                for kind in ('lr', 'mlp'):
                    score[(fs, kind)] = float(np.mean([infer.per_round_accuracy(
                        infer.per_round_fit(tr, kind, C[fs], cfg), va, K) for tr, va in inner[fs]]))
            fs, kind = max(score, key=lambda c: (score[c], c[1] == 'lr'))

            # A or B per k on the same inner folds
            inner_acc = {('A', k): [] for k in ks} | {('B', k): [] for k in ks}
            for tr, va in inner[fs]:
                mdl = infer.per_round_fit(tr, kind, C[fs], cfg)
                for s in va:
                    first, _ = infer.method_a(infer.per_round_probs(mdl, s, K), ks, rng, 1)
                    for k in ks:
                        inner_acc[('A', k)].append((first[k] == s['y']).mean())
                b = infer.method_b(tr, va, ks, cfg, rng, C=C[fs], K=K)
                for k in ks:
                    for s, (first, _) in zip(va, b[k]):
                        inner_acc[('B', k)].append((first == s['y']).mean())
            choice = {k: 'B' if np.mean(inner_acc[('B', k)]) > np.mean(inner_acc[('A', k)]) else 'A' for k in ks}
            sel_rows.append({'mechanism': m, 'eps': e, 'run_seed': test_seed, 'feature_set': fs, 'classifier': kind,
                             **{f'inner_{a}_{b}': round(v, 4) for (a, b), v in score.items()},
                             **{f'C_{f}': C[f] for f in feats}, **{f'choice_k{k}': choice[k] for k in ks},
                             **{f'inner_{a}_k{k}': round(float(np.mean(v)), 4) for (a, k), v in inner_acc.items()}})

            def add(method, preds_first, preds_rand=None, suffix='_random'):
                y = test['labels']
                for k in ks:
                    c = preds_first[k] == y
                    for v, val in enumerate(c):
                        correct_rows.append((m, e, test_seed, test['info']['vehs'][v], k, method, float(val)))
                    if preds_rand is not None:
                        cr = (preds_rand[k] == y[None]).mean(0)
                        for v, val in enumerate(cr):
                            correct_rows.append((m, e, test_seed, test['info']['vehs'][v], k, method + suffix, float(val)))

            # outer: every combination (method A), the chosen one also as "A"
            for f2 in feats:
                for kd in ('lr', 'mlp'):
                    mdl = infer.per_round_fit(tr_sets[f2], kd, C[f2], cfg)
                    P = infer.per_round_probs(mdl, te_set[f2], K)
                    T, V = P.shape[:2]
                    df = pd.DataFrame(P.reshape(T * V, K), columns=[f'p_{d}' for d in range(K)])
                    df.insert(0, 'true_label', np.tile(test['labels'], T))
                    df.insert(0, 'round', np.repeat(np.arange(1, T + 1), V))
                    df.insert(0, 'veh_id', np.tile(test['info']['vehs'], T))
                    df.insert(0, 'classifier', kd)
                    df.insert(0, 'feature_set', f2)
                    df.insert(0, 'run_seed', test_seed)
                    df.insert(0, 'eps', e)
                    df.insert(0, 'mechanism', m)
                    prob_frames.append(df)
                    first, rand = infer.method_a(P, ks, rng, cfg['random_subset_draws'])
                    add(f'A[{f2}/{kd}]', first)
                    if (f2, kd) == (fs, kind):
                        add('A', first, rand)
            b = infer.method_b(tr_sets[fs], [te_set[fs]], ks, cfg, rng, K=K)
            add('B', {k: b[k][0][0] for k in ks}, {k: b[k][0][1] for k in ks})
            # headline: A or B per k as chosen on the inner folds
            mdl = infer.per_round_fit(tr_sets[fs], kind, C[fs], cfg)
            fa, _ = infer.method_a(infer.per_round_probs(mdl, te_set[fs], K), ks, rng, 1)
            add('headline', {k: (b[k][0][0] if choice[k] == 'B' else fa[k]) for k in ks})

            # shuffled-label control: labels permuted among the vehicles of each training run; one
            # permutation maps whole origins to wrong classes, so average over control_perms permutations
            ca, cb = [], []
            for _ in range(cfg['control_perms']):
                sh = [dict(s, y=rng.permutation(s['y'])) for s in tr_sets[fs]]
                mdl = infer.per_round_fit(sh, kind, C[fs], cfg)
                fc, _ = infer.method_a(infer.per_round_probs(mdl, te_set[fs], K), ks, rng, 1)
                ca.append(fc)
                bc = infer.method_b(sh, [te_set[fs]], ks, cfg, rng, C=C[fs], K=K)
                cb.append({k: bc[k][0][0] for k in ks})
            add('control_A', ca[0], {k: np.stack([c[k] for c in ca]) for k in ks}, '_avg')
            add('control_B', cb[0], {k: np.stack([c[k] for c in cb]) for k in ks}, '_avg')
            print(f'attack {m} eps={e} test seed {test_seed}: {fs}/{kind}, choice {choice}, '
                  f'{time.time() - t0:.0f}s', flush=True)

    out = cfg['out_dir']
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(correct_rows, columns=['mechanism', 'eps', 'run_seed', 'veh_id', 'k', 'method', 'correct']) \
        .to_csv(os.path.join(out, 'per_vehicle_correct.csv'), index=False)
    pd.concat(prob_frames).to_csv(os.path.join(out, 'per_round_probs.csv.gz'), index=False, float_format='%.5g')
    pd.DataFrame(sel_rows).to_csv(os.path.join(out, 'selection.csv'), index=False)


def stage_known(cfg):
    """Attack 1 (attack/known.py): known vehicles inside the same run, every finished run."""
    from joblib import Parallel, delayed
    dirs = [d for ds in configs(cfg).values() for d in ds]
    t0 = time.time()
    parts = Parallel(n_jobs=cfg['n_jobs'], verbose=5)(delayed(known.run_one)(d, cfg) for d in dirs)
    rows = [x for p in parts for x in p]
    pd.DataFrame(rows, columns=['mechanism', 'eps', 'run_seed', 'veh_id', 'known_frac', 'k', 'method', 'correct']) \
        .to_csv(os.path.join(cfg['out_dir'], 'per_vehicle_correct_known.csv'), index=False)
    print(f'known: {len(dirs)} runs, {len(rows)} rows, {time.time() - t0:.0f}s', flush=True)


def cluster_bootstrap(df, n, rng):
    """95% CI of mean accuracy: resample runs, then vehicles within each resampled run."""
    by_run = [g['correct'].to_numpy() for _, g in df.groupby('run_seed')]
    stats = np.empty(n)
    for b in range(n):
        picks = rng.integers(0, len(by_run), len(by_run))
        stats[b] = np.mean(np.concatenate([by_run[p][rng.integers(0, len(by_run[p]), len(by_run[p]))] for p in picks]))
    return np.percentile(stats, [2.5, 97.5])


def n_origins(cfg):
    runs = [r for rs in dataset.finished_runs(cfg['exp_dir']).values() for r in rs]
    return len({l for _, l in dataset.load_agents(runs[0])})


def stage_aggregate(cfg):
    rng = np.random.default_rng(cfg['seed'] + 1)
    out = cfg['out_dir']
    d = pd.read_csv(os.path.join(out, 'per_vehicle_correct.csv'))
    d['eps'] = d['eps'].fillna(np.inf)
    rows = []
    for (m, e, k, meth), g in d.groupby(['mechanism', 'eps', 'k', 'method']):
        lo, hi = cluster_bootstrap(g, cfg['bootstrap'], rng)
        rows.append({'mechanism': m, 'eps': e, 'k': k, 'method': meth, 'acc_mean': g['correct'].mean(),
                     'ci_low': lo, 'ci_high': hi, 'n_vehicles': len(g), 'n_runs': g['run_seed'].nunique()})
    f = pd.DataFrame(rows)
    f.to_csv(os.path.join(out, 'fig_rounds_data.csv'), index=False, float_format='%.4f')
    # seed-level accuracy (one row per held-out seed), next to the pooled values
    d.groupby(['mechanism', 'eps', 'k', 'method', 'run_seed'])['correct'].agg(['mean', 'count']).reset_index() \
        .rename(columns={'mean': 'acc', 'count': 'n_vehicles'}) \
        .to_csv(os.path.join(out, 'per_seed_rounds.csv'), index=False, float_format='%.4f')
    h = f[(f.method == 'headline') & f.k.isin([1, max(cfg['ks'])])]
    with open(os.path.join(out, 'attack_table.tex'), 'w') as t:
        t.write(f'% generated by attack/run.py --stage aggregate; top-1 origin accuracy, chance {1 / n_origins(cfg):.2f}, 95% CI\n')
        t.write('\\begin{tabular}{llcc}\n\\toprule\nMechanism & $\\varepsilon$ & $k=1$ & $k=%d$ \\\\\n\\midrule\n' % max(cfg['ks']))
        for (m, e), g in h.groupby(['mechanism', 'eps']):
            cells = [f"{r.acc_mean:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" for r in g.sort_values('k').itertuples()]
            t.write(f"{NAMES.get(m, m)} & {'--' if np.isinf(e) else f'{e:g}'} & " + ' & '.join(cells) + ' \\\\\n')
        t.write('\\bottomrule\n\\end{tabular}\n')
    print(f'{len(f)} rows -> fig_rounds_data.csv, attack_table.tex', flush=True)

    path = os.path.join(out, 'per_vehicle_correct_known.csv')
    if os.path.exists(path):
        d = pd.read_csv(path)
        d['eps'] = d['eps'].fillna(np.inf)
        rows = []
        for (m, e, fr, k, meth), g in d.groupby(['mechanism', 'eps', 'known_frac', 'k', 'method']):
            lo, hi = cluster_bootstrap(g, cfg['bootstrap'], rng)
            rows.append({'mechanism': m, 'eps': e, 'known_frac': fr, 'k': k, 'method': meth, 'acc_mean': g['correct'].mean(),
                         'ci_low': lo, 'ci_high': hi, 'n_vehicles': len(g), 'n_runs': g['run_seed'].nunique()})
        pd.DataFrame(rows).to_csv(os.path.join(out, 'fig_known_data.csv'), index=False, float_format='%.4f')
        d.groupby(['mechanism', 'eps', 'known_frac', 'k', 'method', 'run_seed'])['correct'].agg(['mean', 'count']) \
            .reset_index().rename(columns={'mean': 'acc', 'count': 'n_vehicles'}) \
            .to_csv(os.path.join(out, 'per_seed_known.csv'), index=False, float_format='%.4f')
        print(f'{len(rows)} rows -> fig_known_data.csv', flush=True)


# paper figure colours + no privacy in gray
NAMES = {'none': 'No privacy', 'cpq': 'CPQ', 'ldpfl': 'LDP-FL', 'gaussian': 'Gaussian', 'laplace': 'Laplace'}
COLORS = {'none': '#7f7f7f', 'cpq': '#0072B2', 'ldpfl': '#CC79A7', 'gaussian': '#D55E00', 'laplace': '#009E73'}
MARKERS = {'none': 'x', 'cpq': 'o', 'ldpfl': 'D', 'gaussian': 's', 'laplace': '^'}


def plot_rounds(cfg, f, name, title=None):
    """Accuracy vs rounds observed, one panel per eps, one line per mechanism (f: rows of one method)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    epss = sorted(e for e in f.eps.unique() if np.isfinite(e)) or [None]     # None: no-privacy runs only
    chance = 1 / n_origins(cfg)
    fig, axes = plt.subplots(1, len(epss), figsize=(7.16, 2.2), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], epss):
        for m in ('none', 'gaussian', 'laplace', 'ldpfl', 'cpq'):
            g = f[(f.mechanism == m) & ((f.eps == e) | (m == 'none'))].sort_values('k')
            if g.empty:
                continue
            ax.plot(g.k, g.acc_mean, color=COLORS[m], marker=MARKERS[m], ms=3.5, lw=1.2, label=NAMES[m])
            ax.fill_between(g.k, g.ci_low, g.ci_high, color=COLORS[m], alpha=0.15, lw=0)
        ax.axhline(chance, ls='--', color='k', lw=0.8, label='random guess')
        ax.set_xscale('log')
        ax.set_xticks(cfg['ks'])
        ax.set_xticklabels([str(k) for k in cfg['ks']])
        ax.set_ylim(0, 1)
        ax.set_title('no privacy only (so far)' if e is None else f'$\\varepsilon$ = {e:g}')
        ax.set_xlabel('rounds observed $k$')
        ax.grid(axis='y', color='#e1e1e1', lw=0.45)
    axes[0][0].set_ylabel('origin inference accuracy')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.04))
    if title:
        fig.text(0.01, 0.01, title, fontsize=6.5, color='#555555')
    fig.tight_layout(rect=(0, 0.03 if title else 0, 1, 0.9))
    fig.savefig(os.path.join(cfg['fig_dir'], name + '.pdf'))
    fig.savefig(os.path.join(cfg['fig_dir'], name + '.png'), dpi=200)
    plt.close(fig)


def stage_plot(cfg, method='headline'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(cfg['fig_dir'], exist_ok=True)
    f = pd.read_csv(os.path.join(cfg['out_dir'], 'fig_rounds_data.csv'))
    f = f[f.method == method]
    epss = sorted(e for e in f.eps.unique() if np.isfinite(e))
    chance = 1 / n_origins(cfg)
    plot_rounds(cfg, f, 'attack_vs_rounds', 'shadow runs (other seeds), function- or weight-space features chosen on training runs')
    path = os.path.join(cfg['out_dir'], 'fig_known_data.csv')
    if os.path.exists(path):
        kd = pd.read_csv(path)
        K = n_origins(cfg)
        for fr in sorted(kd.known_frac.unique()):
            per = int(round(fr * 50 / K))
            plot_rounds(cfg, kd[(kd.known_frac == fr) & (kd.method == 'headline')], f'attack_known{int(round(fr * 100))}_vs_rounds',
                        f'known vehicles: {per} per origin = {fr:.0%} of the vehicles of the same run; chance 1/{K}')

    if not epss:
        print('figures ->', cfg['fig_dir'], '(no private runs yet: attack_vs_eps skipped)', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True)
    for ax, k in zip(axes, (1, max(cfg['ks']))):
        for m in ('gaussian', 'laplace', 'ldpfl', 'cpq'):
            g = f[(f.mechanism == m) & (f.k == k)].sort_values('eps')
            if not g.empty:
                ax.errorbar(g.eps, g.acc_mean, yerr=[g.acc_mean - g.ci_low, g.ci_high - g.acc_mean], color=COLORS[m],
                            marker=MARKERS[m], ms=3.5, lw=1.2, capsize=2, label=NAMES[m])
        n = f[(f.mechanism == 'none') & (f.k == k)]
        if not n.empty:
            ax.axhline(n.acc_mean.iloc[0], color=COLORS['none'], lw=1, label=NAMES['none'])
        ax.axhline(chance, ls='--', color='k', lw=0.8, label='random guess')
        ax.set_xscale('log')
        ax.set_xticks(epss)
        ax.set_xticklabels([f'{e:g}' for e in epss])
        ax.set_ylim(0, 1)
        ax.set_title(f'k = {k}')
        ax.set_xlabel('$\\varepsilon$')
    axes[0].set_ylabel('accuracy')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(os.path.join(cfg['fig_dir'], 'attack_vs_eps.pdf'))
    fig.savefig(os.path.join(cfg['fig_dir'], 'attack_vs_eps.png'), dpi=200)
    print('figures ->', cfg['fig_dir'], flush=True)


def stage_report(cfg):
    """REPORT_TABLES.md: per (method, eps) the headline accuracy of both attacks over k (pooled and per seed), the
    shuffled-label controls, the share of updates the server received, and the utility (rounds 91-100)."""
    out = cfg['out_dir']
    K = n_origins(cfg)
    ks = cfg['ks']
    L = [f'# Origin inference attack: tables ({cfg["exp_dir"]})', '',
         f'Generated {time.strftime("%Y-%m-%d %H:%M")}. {K} origins, chance {1 / K:.0%}. Accuracy = share of target '
         f'vehicles whose origin is guessed right after observing rounds 1..k (95% CI: cluster bootstrap over seeds, '
         f'then vehicles).', '']

    def name(m, e):
        return NAMES[m] + ('' if not np.isfinite(e) else f' ε = {e:g}')

    def table(df, title, method, extra=None):
        d = df[df.method == method]
        if d.empty:
            return
        L.extend([f'### {title}', '', '| Method | ' + ' | '.join(f'k = {k}' for k in ks) + ' |',
                  '|---|' + '---|' * len(ks)])
        for (m, e), g in d.groupby(['mechanism', 'eps']):
            g = g.set_index('k')
            L.append(f'| {name(m, e)} | ' + ' | '.join(
                f'{g.acc_mean[k]:.0%} [{g.ci_low[k]:.0%}, {g.ci_high[k]:.0%}]' if k in g.index else '–' for k in ks) + ' |')
        L.append('')

    def per_seed(path, method, title, frac=None):
        if not os.path.exists(path):
            return
        s = pd.read_csv(path)
        s['eps'] = s['eps'].fillna(np.inf)
        s = s[(s.method == method) & s.k.isin([10, 50, max(ks)])]
        if frac is not None:
            s = s[np.isclose(s.known_frac, frac)]
        if s.empty:
            return
        seeds = sorted(s.run_seed.unique())
        L.extend([f'### {title}', '', '| Method | k | ' + ' | '.join(f'seed {x}' for x in seeds) + ' |',
                  '|---|---|' + '---|' * len(seeds)])
        for (m, e, k), g in s.groupby(['mechanism', 'eps', 'k']):
            g = g.set_index('run_seed')
            L.append(f'| {name(m, e)} | {k} | ' + ' | '.join(f'{g.acc[x]:.0%}' if x in g.index else '–' for x in seeds) + ' |')
        L.append('')

    kp = os.path.join(out, 'fig_known_data.csv')
    if os.path.exists(kp):
        kd = pd.read_csv(kp)
        for fr in sorted(kd.known_frac.unique()):
            per = int(round(fr * 50 / K))
            L.extend([f'## Attack 1: {per} known vehicle(s) per origin ({fr:.0%} of the vehicles), same run', ''])
            table(kd[kd.known_frac == fr], 'Headline', 'headline')
            table(kd[kd.known_frac == fr], 'Shuffled-label control (expected about chance)', 'control')
            per_seed(os.path.join(out, 'per_seed_known.csv'), 'headline', 'Attack 1 headline per seed', fr)
    fd = pd.read_csv(os.path.join(out, 'fig_rounds_data.csv'))
    L.extend(['## Attack 2: no known vehicles, shadow runs (train on the other seeds)', ''])
    table(fd, 'Headline (features and A / B chosen on the training seeds)', 'headline')
    table(fd, 'Probe features, method A, logistic regression', 'A[probe/lr]')
    table(fd, 'Shuffled-label control (expected about chance)', 'control_A_avg')
    per_seed(os.path.join(out, 'per_seed_rounds.csv'), 'headline', 'Attack 2 headline per seed')

    L.extend(['## Server view and utility per run', '',
              '| Run | Updates received (share of vehicle-rounds) | Travel time rounds 91–100 (s) | Arrival rounds 91–100 |',
              '|---|---|---|---|'])
    for (m, e), runs in sorted(dataset.finished_runs(cfg['exp_dir']).items(), key=lambda x: (x[0][0], x[0][1] or 0)):
        for r in runs:
            obs = np.load(os.path.join(out, 'cache', features.run_key(r), 'observed.npy')) \
                if os.path.exists(os.path.join(out, 'cache', features.run_key(r), 'observed.npy')) else None
            gp = os.path.join(r, 'global_eval.csv')
            ge = pd.read_csv(gp).set_index('round').loc[91:100] if os.path.exists(gp) else None
            L.append(f'| {os.path.basename(r)} | {"–" if obs is None else f"{obs.mean():.0%}"} | '
                     + ('– | – |' if ge is None else f'{ge.travel_time_mean.mean():.0f} | {ge.arrival_rate.mean():.0%} |'))
    path = os.path.join(out, 'REPORT_TABLES.md')
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('report ->', path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['features', 'test', 'attack', 'known', 'aggregate', 'plot', 'report', 'all'])
    p.add_argument('--config', default='attack/config.yaml')
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    stages = ['features', 'test', 'attack', 'known', 'aggregate', 'plot', 'report'] if args.stage == 'all' else [args.stage]
    for s in stages:
        globals()[f'stage_{s}'](cfg)


if __name__ == '__main__':
    main()
