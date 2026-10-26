"""Attack 1, known vehicles: inside one run, the server knows the origin of some vehicles from side information
(honest members whose home is public; the server does not hold their keys, so not the CPQ group secret) and infers
the origin of every other vehicle of the same run. All vehicles start each round from the same global model, so
their updates are directly comparable (this is what fails across runs: independently initialised networks have
unaligned hidden units).

Known vehicles: known_per_origin vehicles of every origin (e.g. 1 per origin = 10 of 50 vehicles with 10 origins),
known_draws random choices per run.

Online: after k rounds the attacker uses rounds 1..k only, of the known vehicles (training) and of the target
vehicles (prediction), and only the updates it actually received (ns-3: undelivered updates are missing). Labels of
target vehicles are used only for scoring; every target vehicle is scored, also one never received (random guess).

Features per (round, vehicle), centred per round (minus the mean over the vehicles received in that round, which
the server can compute; removes what all vehicles share):
  pos    fc1 weights of the state inputs 90-93 (600 values)
  full   the whole update, PCA fitted on the received updates of rounds 1..k (no labels needed)
  probe  Q-value change on the 60 probe states (180 values)
Methods: A = per-round L2 logistic regression, log-probabilities summed over the received rounds in 1..k;
B = logistic regression on the average of the received rounds in 1..k (trained on the known vehicles' averages
over rounds 1..k and over random subsets of them). Headline variant per (draw, k): chosen on the known vehicles only
(two balanced halves, train on one and score the other, both ways); with one known vehicle per origin no split is
possible and the headline is fixed in advance to cfg['known_default'] (A with probe, the best variant on average at
one known vehicle per origin in the 5-origin ideal-network data, which is independent of the ns-3 runs).
Control: the headline variant trained with the known vehicles' labels permuted, averaged over control_perms
permutations (expected accuracy 1/K).
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from attack import features
from attack.infer import P_MIN, finite_rows, nanmean0, tie_break

VARIANTS = [(fs, m) for fs in ('pos', 'full', 'probe') for m in ('A', 'B')]


def _lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _predict(X, y, train_v, test_v, method, K, C, rng, b_draws):
    """Predictions (len(test_v),) for vehicles test_v from X (k, V, F; NaN = not received) with labels y of train_v."""
    k, F = X.shape[0], X.shape[2]
    if method == 'A':
        Xt = X[:, train_v].reshape(-1, F)
        yt = np.tile(y[train_v], k)
        ok = finite_rows(Xt)
        L = np.zeros((k * len(test_v), K))
        if len(np.unique(yt[ok])) >= 2:
            m = _lr(C).fit(Xt[ok], yt[ok])
            Xs = X[:, test_v].reshape(-1, F)
            ts = finite_rows(Xs)
            if ts.any():
                P = np.full((ts.sum(), K), P_MIN)
                P[:, m.classes_] = m.predict_proba(Xs[ts])
                L[ts] = np.log(np.clip(P, P_MIN, 1))
        return tie_break(L.reshape(k, len(test_v), K).sum(0), rng)
    subsets = [np.arange(k)] + ([np.sort(rng.choice(k, max(1, k // 2), replace=False)) for _ in range(b_draws)]
                                if k > 1 else [])
    Xt, yt = [], []
    for s in subsets:
        M = nanmean0(X[s][:, train_v])
        ok = finite_rows(M)
        Xt.append(M[ok])
        yt.append(y[train_v][ok])
    Xt, yt = np.concatenate(Xt), np.concatenate(yt)
    out = rng.integers(0, K, len(test_v))
    if len(np.unique(yt)) < 2:
        return out
    m = _lr(C).fit(Xt, yt)
    M = nanmean0(X[:, test_v])
    ok = finite_rows(M)
    if ok.any():
        out[ok] = m.predict(M[ok])
    return out


def _balanced(y, per_origin, rng):
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_origin, replace=False)
                                   for c in np.unique(y)]))


def run_one(cache_dir, cfg):
    """Rows (mechanism, eps, run_seed, veh_id, known_frac, k, variant, correct averaged over draws)."""
    r = features.load(cache_dir)
    info, y = r['info'], r['labels']
    T, V = r['pos'].shape[:2]
    K = len(np.unique(y))
    rng = np.random.default_rng(cfg['seed'] * 1000 + info['seed'])
    C, ks, b_draws = cfg['known_C'], cfg['ks'], cfg['b_draws']
    default = tuple(cfg.get('known_default', ['full', 'A']))
    pers = cfg.get('known_per_origin') or [max(1, int(round(f * V / K))) for f in cfg['known_fracs']]

    def centred(X):
        with np.errstate(invalid='ignore'):
            return X - np.nanmean(X, 1, keepdims=True)

    base = {'pos': centred(r['pos']), 'probe': centred(r['probe'])}
    full_k = {}
    for k in ks:          # PCA of the received updates of rounds 1..k, centred per round
        Z = np.asarray(r['full'][:k * V], np.float32).reshape(k, V, -1)[:, r['perm']]
        ok = r['observed'][:k]
        Z -= np.nanmean(np.where(ok[..., None], Z, np.nan), 1, keepdims=True)
        n = min(cfg['known_pca_dim'], int(ok.sum()) - 1)
        out = np.full((k, V, max(n, 1)), np.nan, np.float32)
        if n >= 1:
            out[ok] = PCA(n_components=n, svd_solver='randomized', random_state=cfg['seed']).fit_transform(Z[ok])
        full_k[k] = out
        del Z

    acc = {}              # (frac, k, variant) -> (sum of correct (V,), count (V,))
    for per in pers:
        f = per * K / V
        for _ in range(cfg['known_draws']):
            known = _balanced(y, per, rng)
            target = np.setdiff1d(np.arange(V), known)
            for k in ks:
                X = {'pos': base['pos'][:k], 'probe': base['probe'][:k], 'full': full_k[k]}
                preds = {(fs, m): _predict(X[fs], y, known, target, m, K, C, rng, b_draws) for fs, m in VARIANTS}
                choice = default
                if per >= 2:      # validation on the known vehicles only
                    h1 = _balanced(y[known], per // 2, rng)
                    halves = (known[h1], np.setdiff1d(known, known[h1]))
                    score = {v: np.mean([(_predict(X[v[0]], y, a, b, v[1], K, C, rng, b_draws) == y[b]).mean()
                                         for a, b in (halves, halves[::-1])]) for v in VARIANTS}
                    choice = max(VARIANTS, key=lambda v: (score[v], v == default))
                preds['headline'] = preds[choice]
                ctrl = []
                for _ in range(cfg['control_perms']):
                    yp = y.copy()
                    yp[known] = rng.permutation(y[known])
                    ctrl.append(_predict(X[choice[0]], yp, known, target, choice[1], K, C, rng, b_draws) == y[target])
                for name, p in preds.items():
                    key = (f, k, name if isinstance(name, str) else f'{name[1]}[{name[0]}]')
                    s, c = acc.setdefault(key, (np.zeros(V), np.zeros(V)))
                    s[target] += (p == y[target])
                    c[target] += 1
                s, c = acc.setdefault((f, k, 'control'), (np.zeros(V), np.zeros(V)))
                s[target] += np.mean(ctrl, 0)
                c[target] += 1
    rows = []
    for (f, k, name), (s, c) in acc.items():
        for v in np.flatnonzero(c):
            rows.append((info['method'], info['eps'], info['seed'], info['vehs'][v], f, k, name, s[v] / c[v]))
    return rows
