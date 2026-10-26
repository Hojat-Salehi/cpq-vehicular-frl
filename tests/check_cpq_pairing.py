# tests/check_cpq_pairing.py
# Checks for per-coordinate CPQ pairing (cpq_pairing.py; privacy.cpq_pairing_granularity: coordinate).
#   python tests/check_cpq_pairing.py        (inside sumo_env, from the repo root)
#
# 1. matching     at every coordinate the clients form a perfect matching (odd n: one unpaired)
# 2. roles        within each pair one client uses V and the other 1 - V
# 3. variation    partners change across coordinates and rounds; partner choice is uniform
# 4. determinism  same seed and round -> identical values; other seed or round -> unrelated
# 5. statistics   marginals/unbiasedness hold, pair errors are negatively correlated per coordinate,
#                 and the aggregate error over n clients is the same as with client pairing and
#                 well below LDP-FL
# 6. server       coordinate mode sends a pairing seed (no pair seeds/roles); aggregate correct
# 7. timing       per-round derivation time for 50 clients x 29,653 coordinates
# step 2, local keys from the group secret (privacy.cpq_pairing: local):
# 8. keys         HKDF gives three distinct keys; they change with secret, epoch or roster
# 9. independent  vehicles computing on their own (no shared cache) get consistent pairs (V sums
#                 to 1 with the partner); local per-client pairing keeps one partner per round
# 10. hidden      values derivable without K_G (server pairing seeds, other secrets) are unrelated
# 11. server      local mode sends no pairing information; MSE equal to the server-seeded version

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

from frl.privacy import cpq_pairing
from frl.privacy import onebit

failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def all_values(seed, t, n, m):
    return np.stack([cpq_pairing.coordinate_values(seed, t, i, n, [(m,)], private_seed=1000 + i)[0]
                     for i in range(n)])


def test_matching_roles_variation():
    for n in (10, 9):
        m = 2000
        P = cpq_pairing.partners(7, 1, n, m)
        V = all_values(7, 1, n, m)
        sym = all(P[P[u, j], j] == u for j in range(m) for u in range(n) if P[u, j] >= 0)
        unpaired = (P < 0).sum(axis=0)
        roles = all(abs(V[u, j] + V[P[u, j], j] - 1.0) < 1e-12 for j in range(0, m, 7) for u in range(n) if P[u, j] >= 0)
        check(f'n={n}: perfect matching at every coordinate, partner V sums to 1 (V / 1-V)',
              sym and set(unpaired) == {n % 2} and roles,
              f'unpaired per coordinate: {sorted(set(unpaired))}')
    n, m = 10, 20000
    P1, P2 = cpq_pairing.partners(7, 1, n, m), cpq_pairing.partners(7, 2, n, m)
    same_coord = np.mean(P1[0, 1:] == P1[0, 0])
    same_round = np.mean(P1[0] == P2[0])
    counts = np.bincount(P1[0], minlength=n)[1:]
    chi2 = ((counts - m / (n - 1)) ** 2 / (m / (n - 1))).sum()   # df = 8, 99.9% quantile ~ 26.1
    check('partners change across coordinates and rounds, uniformly over the other clients',
          abs(same_coord - 1 / (n - 1)) < 0.01 and abs(same_round - 1 / (n - 1)) < 0.01 and chi2 < 26.1,
          f'same partner as coordinate 0: {same_coord:.3f}, as round 2: {same_round:.3f} (expected {1/(n-1):.3f}), '
          f'chi2 {chi2:.1f} (df 8)')


def test_determinism():
    a = all_values(7, 3, 6, 500)
    b = all_values(7, 3, 6, 500)
    c = all_values(8, 3, 6, 500)
    d = all_values(7, 4, 6, 500)
    rel = lambda x, y: abs(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    check('deterministic for the same seed and round, unrelated otherwise',
          np.array_equal(a, b) and rel(a, c) < 0.05 and rel(a, d) < 0.05,
          f'|corr| other seed {rel(a, c):.3f}, other round {rel(a, d):.3f}')


def test_statistics():
    eps, r, n, m, trials = 1.0, 0.03, 10, 400, 3000
    a = onebit.alpha(eps)
    rng = np.random.default_rng(0)
    G = [rng.normal(0, 0.1, m)]
    W = [[G[0] + rng.uniform(-r, r, m)] for _ in range(n)]
    true_mean = np.mean([w[0] for w in W], axis=0)

    def aggregate(mode, t):
        dec = []
        for i in range(n):
            if mode == 'coordinate':
                V = cpq_pairing.coordinate_values(11, t, i, n, [(m,)], 10 ** 6 + t * n + i)
                packed, _ = onebit.encode(W[i], G, [r], eps, 'cpq', 0, shared_V=V)
            elif mode == 'client':
                pair, role = i // 2, i % 2
                packed, _ = onebit.encode(W[i], G, [r], eps, 'cpq', t * 1000 + pair, role)
            else:
                packed, _ = onebit.encode(W[i], G, [r], eps, 'ldpfl', t * 1000 + i)
            dec.append(onebit.decode(packed, G, [r], eps)[0])
        return np.mean(dec, axis=0), dec

    out = {}
    for mode in ('coordinate', 'client', 'ldpfl'):
        aggs, d0 = [], []
        for t in range(trials):
            agg, dec = aggregate(mode, t)
            aggs.append(agg)
            d0.append(dec[0])
        aggs = np.array(aggs)
        out[mode] = (np.mean((aggs - true_mean) ** 2), np.max(np.abs(aggs.mean(0) - true_mean)) /
                     (a * r / np.sqrt(n * trials)), np.max(np.abs(np.mean(d0, 0) - W[0][0])) / (a * r / np.sqrt(trials)))
    check('coordinate pairing: unbiased per client and in the aggregate (within 5 standard errors)',
          out['coordinate'][1] < 5 and out['coordinate'][2] < 5,
          f"aggregate {out['coordinate'][1]:.1f} SE, client 0 {out['coordinate'][2]:.1f} SE")
    ratio_client = out['coordinate'][0] / out['client'][0]
    ratio_ldp = out['coordinate'][0] / out['ldpfl'][0]
    check('coordinate pairing: aggregate MSE equal to client pairing, well below LDP-FL',
          abs(ratio_client - 1) < 0.1 and ratio_ldp < 0.3,
          f"MSE coordinate / client = {ratio_client:.2f}, coordinate / LDP-FL = {ratio_ldp:.2f}")


def test_server():
    import flwr as fl
    from flwr.common import Code, FitRes, GetPropertiesRes, Status
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy
    from frl.federated.server_onebit import OneBitStrategy

    class Proxy(ClientProxy):
        def get_properties(self, ins, timeout, group_id):
            return GetPropertiesRes(status=Status(Code.OK, ''), properties={'veh': f'veh{self.cid}'})
        get_parameters = fit = evaluate = reconnect = lambda *a, **k: None

    os.environ['FL_PQC_ENABLED'] = os.environ['FL_ORACLE_ENABLED'] = '0'
    shapes = [(94, 150), (150,), (150, 100), (100,), (100, 3), (3,)]
    rng = np.random.default_rng(1)
    g = [rng.normal(0, 0.1, s).astype(np.float32) for s in shapes]
    radius = 0.02
    st = OneBitStrategy('cpq', 1.0, radius, 1.0, fraction_fit=1.0, min_fit_clients=5, min_available_clients=5,
                        pairing_granularity='coordinate', pairing_seed=99)
    cm = SimpleClientManager()
    for i in range(5):
        cm.register(Proxy(str(i)))
    ins = st.configure_fit(3, fl.common.ndarrays_to_parameters(g), cm)
    cfgs = [i.config for _, i in ins]
    cfg_ok = all(c.get('pairing') == 'coordinate' and c.get('pairing_seed') == 99 and c.get('n_clients') == 5
                 and 'pair_seed' not in c for c in cfgs) and sorted(c['client_idx'] for c in cfgs) == list(range(5))
    results, decoded = [], []
    for c, i in ins:
        w = [x + rng.normal(0, 0.01, x.shape).astype(np.float32) for x in g]
        V = cpq_pairing.coordinate_values(99, 3, i.config['client_idx'], 5, shapes, 5000 + i.config['client_idx'])
        packed, _ = onebit.encode(w, g, [radius] * 6, 1.0, 'cpq', 0, shared_V=V)
        decoded.append(onebit.decode(packed, g, [radius] * 6, 1.0))
        results.append((c, FitRes(status=Status(Code.OK, ''), parameters=fl.common.ndarrays_to_parameters(packed),
                                  num_examples=1000, metrics={})))
    agg = fl.common.parameters_to_ndarrays(st.aggregate_fit(3, results, [])[0])
    mean = [np.mean([d[k] for d in decoded], axis=0) for k in range(6)]
    check('server (coordinate mode): sends pairing seed, list size and position, no pair seeds; aggregate correct',
          cfg_ok and all(np.allclose(a, b, atol=1e-6) for a, b in zip(agg, mean)))


def test_timing():
    shapes = [(94, 150), (150,), (150, 100), (100,), (100, 3), (3,)]
    t0 = time.time()
    cpq_pairing.coordinate_values(123, 1, 0, 50, shapes, 1)
    first = time.time() - t0
    t0 = time.time()
    for i in range(1, 50):
        cpq_pairing.coordinate_values(123, 1, i, 50, shapes, 1)
    rest = time.time() - t0
    check('timing: 50 clients x 29,653 coordinates per round', first < 5,
          f'first client {first:.2f} s (computes the round), other 49 together {rest:.2f} s')


def test_local_keys():
    roster = [f'veh{k}' for k in range(10)]
    K = cpq_pairing.seeded_group_secret(1)
    k1 = cpq_pairing.group_keys(K, roster)
    ok = (k1 == cpq_pairing.group_keys(K, roster) and len(set(k1)) == 3
          and k1 != cpq_pairing.group_keys(cpq_pairing.seeded_group_secret(2), roster)
          and k1 != cpq_pairing.group_keys(K, roster, epoch=2)
          and k1 != cpq_pairing.group_keys(K, roster[::-1]))
    check('keys: K_pair, K_coin, K_check distinct, deterministic, depend on secret, epoch and roster', ok)


def test_local_independent_and_hidden():
    roster = [f'veh{k}' for k in range(10)]
    kp, kc, _ = cpq_pairing.group_keys(cpq_pairing.seeded_group_secret(1), roster)
    m, n = 3000, 10
    rows = []
    for i in range(n):                       # every vehicle on its own: clear the shared cache
        cpq_pairing._round_values.cache_clear()
        rows.append(cpq_pairing.shared_values(kp, kc, 5, i, n, [(m,)], 100 + i, True)[0])
    V = np.stack(rows)
    P = cpq_pairing.partners(None, 5, n, m, keys=(kp, kc))
    consistent = all(abs(V[u, j] + V[P[u, j], j] - 1) < 1e-12 for j in range(0, m, 5) for u in range(n))
    check('independent derivation: every vehicle gets values consistent with its partner (V + V_partner = 1)',
          consistent)
    Pc1 = cpq_pairing.partners(None, 5, n, m, keys=(kp, kc), per_coordinate=False)
    Pc2 = cpq_pairing.partners(None, 6, n, m, keys=(kp, kc), per_coordinate=False)
    check('local per-client pairing: one partner for all coordinates, a new one next round',
          bool((Pc1 == Pc1[:, :1]).all()) and not np.array_equal(Pc1[:, 0], Pc2[:, 0]))
    server_view = np.stack([cpq_pairing.coordinate_values(s, 5, 0, n, [(m,)], 1)[0] for s in range(20)])
    other_secret = cpq_pairing.shared_values(*cpq_pairing.group_keys(cpq_pairing.seeded_group_secret(2), roster)[:2],
                                             5, 0, n, [(m,)], 1, True)[0]
    worst = max(abs(np.corrcoef(V[0], x)[0, 1]) for x in list(server_view) + [other_secret])
    check('hidden: values derivable without K_G (20 server pairing seeds, another secret) are unrelated',
          worst < 0.07, f'max |corr| {worst:.3f}')


def test_local_server_and_mse():
    import flwr as fl
    from flwr.common import Code, GetPropertiesRes, Status
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy
    from frl.federated.server_onebit import OneBitStrategy

    class Proxy(ClientProxy):
        def get_properties(self, ins, timeout, group_id):
            return GetPropertiesRes(status=Status(Code.OK, ''), properties={'veh': f'veh{self.cid}'})
        get_parameters = fit = evaluate = reconnect = lambda *a, **k: None

    os.environ['FL_PQC_ENABLED'] = os.environ['FL_ORACLE_ENABLED'] = '0'
    st = OneBitStrategy('cpq', 1.0, 0.02, 1.0, fraction_fit=1.0, min_fit_clients=4, min_available_clients=4,
                        pairing_granularity='coordinate', pairing_mode='local')
    cm = SimpleClientManager()
    for i in range(4):
        cm.register(Proxy(str(i)))
    ins = st.configure_fit(2, fl.common.ndarrays_to_parameters([np.zeros(3, np.float32)]), cm)
    leaked = [k for _, i in ins for k in i.config if k in ('pair_seed', 'role', 'pairing_seed', 'pairing', 'n_clients')]
    check('server (local mode): sends no pairing information', not leaked, f'fields {sorted(set(leaked))}')

    eps, r, n, m, trials = 1.0, 0.03, 10, 300, 2000
    rng = np.random.default_rng(0)
    G = [rng.normal(0, 0.1, m)]
    W = [[G[0] + rng.uniform(-r, r, m)] for _ in range(n)]
    true_mean = np.mean([w[0] for w in W], axis=0)
    roster = [f'veh{k}' for k in range(n)]
    kp, kc, _ = cpq_pairing.group_keys(cpq_pairing.seeded_group_secret(3), roster)
    mse = {}
    for mode in ('local', 'server'):
        errs = []
        for t in range(trials):
            dec = []
            for i in range(n):
                V = (cpq_pairing.shared_values(kp, kc, t, i, n, [(m,)], t * n + i) if mode == 'local'
                     else cpq_pairing.coordinate_values(11, t, i, n, [(m,)], t * n + i))
                dec.append(onebit.decode(onebit.encode(W[i], G, [r], eps, 'cpq', 0, shared_V=V)[0], G, [r], eps)[0])
            errs.append(np.mean(dec, axis=0) - true_mean)
        mse[mode] = np.mean(np.square(errs))
    check('local keys: aggregate MSE equal to the server-seeded pairing',
          abs(mse['local'] / mse['server'] - 1) < 0.1, f"ratio {mse['local'] / mse['server']:.2f}")


if __name__ == '__main__':
    test_matching_roles_variation()
    test_determinism()
    test_statistics()
    test_server()
    test_timing()
    test_local_keys()
    test_local_independent_and_hidden()
    test_local_server_and_mse()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
