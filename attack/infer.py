"""The attack: per-round classifiers, multi-round aggregation (methods A and B), leave-one-run-out.

A "set" is one run's samples restricted to some vehicles: X (T, V, F) unscaled features, y (V,) labels.
Training uses only training sets; the held-out run is touched only by predict / scoring.

Method A: per-round classifier p_t(d | u_t); prediction after k rounds = argmax_d sum_{t<=k} log p_t(d).
Method B: features averaged over k rounds; a classifier trained on k-averaged features of the training runs
          (first k rounds plus random k-round subsets) predicts from the first-k average.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

P_MIN = 1e-6


def finite_rows(X):
    """Rows of the last axis without NaN (NaN = update not received by the server)."""
    return np.isfinite(X).all(-1)


def tie_break(L, rng):
    """Argmax over the last axis with random tie-breaking (a vehicle with no received update in rounds 1..k has
    equal scores for every origin: the attacker guesses at random)."""
    return (L + 1e-9 * rng.random(L.shape)).argmax(-1)


def nanmean0(X):
    """Mean over axis 0 ignoring NaN; NaN where nothing was observed (no warning)."""
    n = np.isfinite(X).sum(0)
    s = np.nansum(X, 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


# ---------- features ----------
def pca_project(train_runs, all_runs, cfg, rng):
    """F_full: PCA fitted on rows of the training runs only; returns {key: (T, V, pca_dim)}."""
    rows = []
    per = max(1, cfg['pca_fit_rows'] // len(train_runs))
    for r in train_runs:
        ok = np.flatnonzero(r['observed'][:, np.argsort(r['perm'])].ravel())   # rows t * V + v received
        idx = np.sort(rng.choice(ok, size=min(per, len(ok)), replace=False))
        rows.append(np.asarray(r['full'][idx], np.float32))
    pca = PCA(n_components=cfg['pca_dim'], svd_solver='randomized', random_state=cfg['seed']).fit(np.concatenate(rows))
    out = {}
    for r in all_runs:
        T, V = r['pos'].shape[:2]
        Z = np.full((T * V, cfg['pca_dim']), np.nan)
        for i in range(0, T * V, 2000):
            blk = np.asarray(r['full'][i:i + 2000], np.float32)
            ok = finite_rows(blk)
            if ok.any():
                Z[i:i + 2000][ok] = pca.transform(blk[ok])
        out[r['info']['run_dir']] = Z.reshape(T, V, -1)[:, r['perm']].astype(np.float32)
    return out


# ---------- classifiers ----------
def _flatten(sets):
    """Received (round, vehicle) samples of the sets."""
    X = np.concatenate([s['X'].reshape(-1, s['X'].shape[2]) for s in sets])
    y = np.concatenate([np.tile(s['y'], s['X'].shape[0]) for s in sets])
    g = np.concatenate([np.tile(s['gid'], s['X'].shape[0]) for s in sets])
    ok = finite_rows(X)
    return X[ok], y[ok], g[ok]


def _transfer_groups(groups):
    """Validation must mimic the attack: with 2+ training runs, hold out whole runs (a classifier that only works
    inside one run, e.g. on raw weights, must not be selected); with one training run, hold out vehicles."""
    runs = np.array([str(g).split(':')[0] for g in groups])
    return runs if len(np.unique(runs)) >= 2 else groups


def choose_C(X, y, groups, cfg):
    groups = _transfer_groups(groups)
    folds = min(cfg['inner_folds'], len(np.unique(groups)))
    best, best_acc = None, -1
    for C in cfg['lr_Cs']:
        acc = []
        for tr, va in GroupKFold(folds).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y[tr])
            acc.append(m.score(X[va], y[va]))
        if np.mean(acc) > best_acc:
            best, best_acc = C, np.mean(acc)
    return best


def make_clf(kind, C, cfg):
    if kind == 'lr':
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(cfg['mlp_hidden'],), early_stopping=True,
                                                         validation_fraction=0.15, max_iter=300,
                                                         random_state=cfg['seed']))


def per_round_fit(train_sets, kind, C, cfg):
    X, y, _ = _flatten(train_sets)
    return make_clf(kind, C, cfg).fit(X, y)


def per_round_probs(model, s, K):
    """(T, V, K) probabilities for one set (all K columns, also for labels absent in training); uniform for rounds
    in which the server received nothing from the vehicle."""
    T, V, F = s['X'].shape
    X = s['X'].reshape(-1, F)
    ok = finite_rows(X)
    p = np.full((T * V, K), 1.0 / K)
    if ok.any():
        q = np.full((ok.sum(), K), P_MIN)
        q[:, model.classes_] = model.predict_proba(X[ok])
        p[ok] = q
    return p.reshape(T, V, K)


# ---------- aggregation over rounds ----------
def method_a(P, ks, rng, draws):
    """{k: (correct_first_k (V,) bool predictions, correct_random (V,) mean over draws)} given (T, V, K)."""
    L = np.log(np.clip(P, P_MIN, 1.0))
    T = L.shape[0]
    first = {k: tie_break(L[:k].sum(0), rng) for k in ks}
    rand = {k: np.stack([tie_break(L[np.sort(rng.choice(T, k, replace=False))].sum(0), rng) for _ in range(draws)])
            for k in ks}
    return first, rand


def _b_train(train_sets, k, cfg, rng):
    X, y, g = [], [], []
    for s in train_sets:
        T = s['X'].shape[0]
        subsets = [np.arange(k)] + [np.sort(rng.choice(T, k, replace=False)) for _ in range(cfg['b_draws'] if k < T else 0)]
        for idx in subsets:
            m = nanmean0(s['X'][idx])
            ok = finite_rows(m)
            X.append(m[ok])
            y.append(s['y'][ok])
            g.append(s['gid'][ok])
    return np.concatenate(X), np.concatenate(y), np.concatenate(g)


def _b_predict(model, M, K, rng):
    """Predictions from averaged features M (V, F); random guess where nothing was received."""
    out = rng.integers(0, K, len(M))
    ok = finite_rows(M)
    if ok.any():
        out[ok] = model.predict(M[ok])
    return out


def method_b(train_sets, test_sets, ks, cfg, rng, C=None, K=None):
    """{k: [ (pred_first (V,), pred_random (draws, V)) per test set ]}; LR on k-averaged features."""
    out = {}
    for k in ks:
        X, y, g = _b_train(train_sets, k, cfg, rng)
        c = C if C is not None else choose_C(X, y, g, cfg)
        m = make_clf('lr', c, cfg).fit(X, y)
        res = []
        K = K or int(max(y.max(), *(s['y'].max() for s in test_sets)) + 1)
        for s in test_sets:
            T = s['X'].shape[0]
            first = _b_predict(m, nanmean0(s['X'][:k]), K, rng)
            rand = np.stack([_b_predict(m, nanmean0(s['X'][np.sort(rng.choice(T, k, replace=False))]), K, rng)
                             for _ in range(cfg['random_subset_draws'])])
            res.append((first, rand))
        out[k] = res
    return out


# ---------- inner selection on training runs only ----------
def inner_splits(train_sets, cfg):
    """Folds over the training sets, by run if there are 2+ training runs, else by vehicle:
    [(inner_train_sets, inner_val_sets)]."""
    gids = np.concatenate([s['gid'] for s in train_sets])
    groups = _transfer_groups(gids)
    folds = min(cfg['inner_folds'], len(np.unique(groups)))
    out = []
    for _, va in GroupKFold(folds).split(gids, groups=groups):
        val = set(gids[va])
        tr_sets, va_sets = [], []
        for s in train_sets:
            mv = np.array([gg in val for gg in s['gid']])
            for mask, dest in ((~mv, tr_sets), (mv, va_sets)):
                if mask.any():
                    dest.append({'X': s['X'][:, mask], 'y': s['y'][mask], 'gid': s['gid'][mask]})
        out.append((tr_sets, va_sets))
    return out


def per_round_accuracy(model, sets, K):
    """Accuracy over the received (round, vehicle) samples."""
    acc = []
    for s in sets:
        ok = finite_rows(s['X'])
        acc.append((per_round_probs(model, s, K).argmax(2) == s['y'][None])[ok].mean())
    return float(np.mean(acc))
