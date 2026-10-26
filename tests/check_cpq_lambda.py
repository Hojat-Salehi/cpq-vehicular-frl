# tests/check_cpq_lambda.py
# Checks for CPQ-lambda (onebit.encode with cpq_lambda < 1) on the code path of the runs
# (local per-coordinate pairing from the group keys, cpq_pairing.shared_values):
#   python tests/check_cpq_lambda.py        (inside sumo_env, from the repo root)
#
# 1. marginal     P(bit = 1) = q for every lambda (decoding unbiased)
# 2. lambda = 0   partner bits independent and aggregate MSE equal to LDP-FL (same updates)
# 3. lambda = 1   identical bits to plain CPQ
# 4. trend        aggregate MSE per lambda (0, 0.25, 0.5, 0.75, 0.9, 1), printed

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from frl.privacy import cpq_pairing
from frl.privacy import onebit

failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


EPS, R, N, M, TRIALS = 1.0, 0.02, 50, 2000, 300
rng = np.random.default_rng(0)
G = [rng.normal(0, 0.1, M)]
W = [[G[0] + np.clip(rng.normal(0, 0.012, M), -0.03, 0.03)] for _ in range(N)]     # some coordinates clipped
true_mean = np.mean([G[0] + np.clip(w[0] - G[0], -R, R) for w in W], axis=0)
roster = [f'veh{k}' for k in range(N)]
kp, kc, _ = cpq_pairing.group_keys(cpq_pairing.seeded_group_secret(3), roster)
q = onebit.bit_probability(W[0], G, [R], EPS)[0]


def bits_of(method, lam, t, i):
    own = int(np.random.SeedSequence([1000 + i, t]).generate_state(1)[0])
    if method == 'ldpfl':
        packed = onebit.encode(W[i], G, [R], EPS, 'ldpfl', own)[0]
    else:
        V = cpq_pairing.shared_values(kp, kc, t, i, N, [(M,)], own)
        packed = onebit.encode(W[i], G, [R], EPS, 'cpq', 0, cpq_lambda=lam, private_seed=own, shared_V=V)[0]
    return np.unpackbits(packed[0])[:M]


def run(method, lam):
    errs, b0 = [], []
    for t in range(TRIALS):
        dec = []
        for i in range(N):
            b = bits_of(method, lam, t, i)
            if i == 0:
                b0.append(b)
            dec.append(G[0] + onebit.alpha(EPS) * R * (2.0 * b - 1.0))
        errs.append(np.mean(dec, axis=0) - true_mean)
    return float(np.mean(np.square(errs))), np.array(b0)


mse, freq = {}, {}
for name, method, lam in [('ldpfl', 'ldpfl', None)] + [(f'lam{l:g}', 'cpq', l) for l in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]:
    mse[name], b0 = run(method, lam)
    freq[name] = b0.mean(0)
    print(f'  {name:8s} aggregate MSE {mse[name]:.3e}', flush=True)

se = np.sqrt(q * (1 - q) / TRIALS)
for name, f in freq.items():
    z = np.abs(f - q) / se
    check(f'marginal {name}: P(bit = 1) = q', np.mean(z > 3) < 0.01, f'share of coordinates beyond 3 SE {np.mean(z > 3):.4f}')

# partner bits at lambda = 0: correlation over trials and coordinates
corr = {}
for lam in (0.0, 1.0):
    a, b = [], []
    for t in range(60):
        P = cpq_pairing.partners(None, t, N, M, keys=(kp, kc))
        bj = np.stack([bits_of('cpq', lam, t, j) for j in range(N)])
        ok = P[0] >= 0
        a.append(bj[0][ok])
        b.append(bj[P[0][ok], np.flatnonzero(ok)])
    corr[lam] = float(np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1])
check('lambda = 0: partner bits uncorrelated', abs(corr[0.0]) < 0.02, f'corr {corr[0.0]:+.4f} (lambda = 1: {corr[1.0]:+.4f})')
ratio = mse['lam0'] / mse['ldpfl']
check('lambda = 0: aggregate MSE equal to LDP-FL', abs(ratio - 1) < 0.05, f'ratio {ratio:.3f}')

same = all(np.array_equal(
    onebit.encode(W[i], G, [R], EPS, 'cpq', 0, cpq_lambda=1.0, private_seed=7,
                  shared_V=cpq_pairing.shared_values(kp, kc, 5, i, N, [(M,)], 7))[0][0],
    onebit.encode(W[i], G, [R], EPS, 'cpq', 0, shared_V=cpq_pairing.shared_values(kp, kc, 5, i, N, [(M,)], 7))[0][0])
    for i in range(N))
check('lambda = 1: bits identical to plain CPQ', same)
order = [mse[f'lam{l:g}'] for l in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]
check('MSE decreases from lambda = 0 to lambda = 1', all(x > y for x, y in zip(order, order[1:])),
      ' > '.join(f'{x:.3e}' for x in order))

print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
