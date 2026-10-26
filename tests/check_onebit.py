# tests/check_onebit.py
# Checks for the one-bit private quantizers (onebit.py, server_onebit.py).
#   python tests/check_onebit.py        (inside sumo_env, from the repo root)
#
# 1. marginal     P(bit = 1) = q for LDP-FL and for both CPQ roles (follower uses 1 - V)
# 2. unbiased     the decoded value averages to the clipped weight (CPQ and LDP-FL)
# 3. clipping     updates beyond r decode, on average, to w_g +/- r
# 4. privacy      q stays in [1/2 - 1/(2 alpha), 1/2 + 1/(2 alpha)]: likelihood ratio <= e^eps
# 5. correlation  CPQ pair errors are negatively correlated; pair-average MSE below LDP-FL's
# 6. wire format  encode/decode round trip on the DQN layer shapes, packed sizes
# 7. server       CPQ pairs get opposite roles; aggregate = mean of the decoded models
# 8. CPQ-lambda   lambda = 1 is bit-identical to CPQ; lambda = 0.5 keeps P(bit=1) = q and
#                 unbiasedness; pair error correlation scales with mu = lambda^2; eps_C formula

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

import onebit

failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


EPS, R = 1.0, 0.03
A = onebit.alpha(EPS)
rng = np.random.default_rng(0)
G = [rng.normal(0, 0.1, 400)]                       # global weights
D = rng.uniform(-R, R, 400)                         # updates inside the interval
W = [G[0] + D]
TRIALS = 20000


def decoded_many(method, w, role=1, seeds=None):
    seeds = seeds if seeds is not None else range(TRIALS)
    return np.stack([onebit.decode(onebit.encode(w, G, [R], EPS, method, s, role)[0], G, [R], EPS)[0]
                     for s in seeds])


def test_marginal_unbiased():
    q = onebit.bit_probability(W, G, [R], EPS)[0]
    out = {}
    for name, method, role in (('ldpfl', 'ldpfl', 1), ('cpq leader', 'cpq', 1), ('cpq follower', 'cpq', 0)):
        dec = decoded_many(method, W, role)
        p1 = ((dec - G[0]) > 0).mean(axis=0)                      # empirical P(bit = 1)
        se = np.sqrt(q * (1 - q) / TRIALS)
        out[name] = (np.max(np.abs(p1 - q) / se), np.max(np.abs(dec.mean(0) - W[0]) / (A * R / np.sqrt(TRIALS))))
    check('marginal: P(bit=1) = q for LDP-FL and both CPQ roles (within 5 standard errors)',
          all(v[0] < 5 for v in out.values()), ', '.join(f'{k}: max {v[0]:.1f} SE' for k, v in out.items()))
    check('unbiased: decoded mean = clipped weight (within 5 standard errors)',
          all(v[1] < 5 for v in out.values()), ', '.join(f'{k}: max {v[1]:.1f} SE' for k, v in out.items()))


def test_clipping_privacy():
    big = [G[0] + np.where(np.arange(400) % 2 == 0, 5 * R, -5 * R)]
    dec = decoded_many('ldpfl', big, seeds=range(4000))
    target = G[0] + np.clip(big[0] - G[0], -R, R)
    _, clipped = onebit.encode(big, G, [R], EPS, 'ldpfl', 1)
    check('clipping: updates beyond r decode to w_g +/- r, reported clipped share 100%',
          np.max(np.abs(dec.mean(0) - target)) < 5 * A * R / np.sqrt(4000) and clipped[0] == 1.0)
    q = onebit.bit_probability([G[0] + rng.uniform(-10, 10, 400)], G, [R], EPS)[0]
    lo, hi = 0.5 - 1 / (2 * A), 0.5 + 1 / (2 * A)
    check('privacy: q within [1/2 - 1/(2a), 1/2 + 1/(2a)], ratio <= e^eps',
          q.min() >= lo - 1e-12 and q.max() <= hi + 1e-12 and hi / lo <= np.exp(EPS) + 1e-9,
          f'q in [{q.min():.4f}, {q.max():.4f}], max ratio {hi / lo:.4f} vs e^eps {np.exp(EPS):.4f}')


def test_correlation():
    w2 = [G[0] - D * 0.5]
    lead = decoded_many('cpq', W, 1)
    foll = decoded_many('cpq', w2, 0)
    e1, e2 = lead - W[0], foll - w2[0]
    corr = np.mean([np.corrcoef(e1[:, j], e2[:, j])[0, 1] for j in range(0, 400, 20)])
    mse_cpq = np.mean(((e1 + e2) / 2) ** 2)
    ind1 = decoded_many('ldpfl', W, seeds=range(TRIALS)) - W[0]
    ind2 = decoded_many('ldpfl', w2, seeds=range(TRIALS, 2 * TRIALS)) - w2[0]
    mse_ind = np.mean(((ind1 + ind2) / 2) ** 2)
    check('correlation: CPQ pair errors negatively correlated, pair-average MSE below LDP-FL',
          corr < -0.5 and mse_cpq < 0.8 * mse_ind,
          f'error correlation {corr:.2f}; pair-average MSE CPQ {mse_cpq:.2e} vs LDP-FL {mse_ind:.2e} '
          f'({mse_cpq / mse_ind:.0%})')


def test_wire():
    shapes = [(94, 150), (150,), (150, 100), (100,), (100, 3), (3,)]
    g = [rng.normal(0, 0.1, s).astype(np.float32) for s in shapes]
    w = [x + rng.normal(0, 0.01, x.shape).astype(np.float32) for x in g]
    radii = [0.02] * 6
    packed, _ = onebit.encode(w, g, radii, EPS, 'cpq', 7, 1)
    dec = onebit.decode(packed, g, radii, EPS)
    n_bytes = sum(p.nbytes for p in packed[0::2])
    ok = all(d.shape == x.shape and d.dtype == np.float32 for d, x in zip(dec, g)) and \
        all(np.allclose(np.abs(d - x), A * r, rtol=1e-5) for d, x, r in zip(dec, g, radii))
    check('wire format: round trip on DQN shapes, every value at w_g +/- alpha*r', ok,
          f'{sum(int(np.prod(s)) for s in shapes)} weights -> {n_bytes} bytes of bits')


def test_server():
    import flwr as fl
    from flwr.common import Code, FitRes, GetPropertiesRes, Status
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.client_proxy import ClientProxy
    from server_onebit import OneBitStrategy

    class Proxy(ClientProxy):
        def get_properties(self, ins, timeout, group_id):
            return GetPropertiesRes(status=Status(Code.OK, ''), properties={'veh': f'veh{self.cid}'})
        get_parameters = fit = evaluate = reconnect = lambda *a, **k: None

    os.environ['FL_PQC_ENABLED'] = os.environ['FL_ORACLE_ENABLED'] = '0'
    shapes = [(94, 150), (150,), (150, 100), (100,), (100, 3), (3,)]
    g = [rng.normal(0, 0.1, s).astype(np.float32) for s in shapes]
    radius = {n: 0.02 for n in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc_out.weight', 'fc_out.bias']}
    st = OneBitStrategy('cpq', EPS, radius, 1.0, fraction_fit=1.0, min_fit_clients=6, min_available_clients=6)
    cm = SimpleClientManager()
    for i in range(6):
        cm.register(Proxy(str(i)))
    ins = st.configure_fit(1, fl.common.ndarrays_to_parameters(g), cm)
    cfgs = {c.cid: i.config for c, i in ins}
    by_seed = {}
    for cid, c in cfgs.items():
        by_seed.setdefault(c['pair_seed'], []).append(c['role'])
    roles_ok = all(sorted(r) == [0, 1] for r in by_seed.values()) and len(by_seed) == 3
    results, decoded = [], []
    for c, i in ins:
        w = [x + rng.normal(0, 0.01, x.shape).astype(np.float32) for x in g]
        packed, _ = onebit.encode(w, g, [0.02] * 6, EPS, 'cpq', i.config['pair_seed'], i.config['role'])
        decoded.append(onebit.decode(packed, g, [0.02] * 6, EPS))
        results.append((c, FitRes(status=Status(Code.OK, ''), parameters=fl.common.ndarrays_to_parameters(packed),
                                  num_examples=1000, metrics={})))
    agg, _ = st.aggregate_fit(1, results, [])
    agg = fl.common.parameters_to_ndarrays(agg)
    mean = [np.mean([d[k] for d in decoded], axis=0) for k in range(6)]
    check('server: CPQ pairs get opposite roles, aggregate = mean of decoded models',
          roles_ok and all(np.allclose(a, m, atol=1e-6) for a, m in zip(agg, mean)),
          f'pairs {list(by_seed.values())}')


def test_cpq_lambda():
    same = all(np.array_equal(a, b) for a, b in zip(onebit.encode(W, G, [R], EPS, 'cpq', 5, 0)[0],
                                                     onebit.encode(W, G, [R], EPS, 'cpq', 5, 0, 1.0, 99)[0]))
    check('CPQ-lambda: lambda = 1 gives exactly the CPQ bits', same)

    def dec(w, role, lam, seeds, offset):
        return np.stack([onebit.decode(onebit.encode(w, G, [R], EPS, 'cpq', s, role, lam, s + offset)[0],
                                       G, [R], EPS)[0] for s in seeds])
    q = onebit.bit_probability(W, G, [R], EPS)[0]
    d = dec(W, 1, 0.5, range(TRIALS), 10 ** 7)
    p1 = ((d - G[0]) > 0).mean(0)
    check('CPQ-lambda 0.5: P(bit=1) = q and unbiased (within 5 standard errors)',
          np.max(np.abs(p1 - q) / np.sqrt(q * (1 - q) / TRIALS)) < 5
          and np.max(np.abs(d.mean(0) - W[0]) / (A * R / np.sqrt(TRIALS))) < 5)
    w2 = [G[0] - D * 0.5]
    corr = {}
    for lam in (1.0, 0.5, 0.0):
        e1 = dec(W, 1, lam, range(8000), 10 ** 7) - W[0]
        e2 = dec(w2, 0, lam, range(8000), 2 * 10 ** 7) - w2[0]
        corr[lam] = np.mean([np.corrcoef(e1[:, j], e2[:, j])[0, 1] for j in range(0, 400, 20)])
    ratio = corr[0.5] / corr[1.0]
    check('CPQ-lambda: pair error correlation scales with mu = lambda^2 (0.25 at lambda 0.5, 0 at 0)',
          abs(ratio - 0.25) < 0.06 and abs(corr[0.0]) < 0.05,
          f'correlation lambda 1: {corr[1.0]:.2f}, 0.5: {corr[0.5]:.2f} (ratio {ratio:.2f}), 0: {corr[0.0]:.2f}')
    vals = {lam: onebit.epsilon_seed_aware(1.0, lam) for lam in (0.0, 0.5, 0.9, 1.0)}
    check('CPQ-lambda: eps_C = ln((e^eps + lambda)/(1 - lambda))',
          abs(vals[0.0] - 1.0) < 1e-9 and abs(vals[0.5] - np.log((np.e + 0.5) / 0.5)) < 1e-9
          and vals[1.0] == float('inf'), ', '.join(f'lambda {k}: {v:.2f}' for k, v in vals.items()))


if __name__ == '__main__':
    test_marginal_unbiased()
    test_clipping_privacy()
    test_correlation()
    test_wire()
    test_server()
    test_cpq_lambda()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
