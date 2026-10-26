# tests/check_additive.py
# Checks for the additive-noise baselines (additive.py).
#   python tests/check_additive.py        (inside sumo_env, from the repo root)
#
# 1. calibration  analytic Gaussian sigma meets the paper's condition with equality and is the
#                 smallest such sigma; Laplace b = 2r / eps gives density ratio e^eps
# 2. unbiased     noisy weights average to the clipped weight; noise variance = 2 b^2 / sigma^2
# 3. clipping     updates beyond r are clipped to w_g +/- r before the noise; clipped share reported
# 4. per layer    each layer's noise scale follows its own radius

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

import additive

failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def test_calibration():
    rows, ok = [], True
    for eps in (1.0, 3.0, 10.0):
        s = additive.analytic_sigma(eps, 1e-5, 2.0)
        at = additive._delta_of(s, eps, 2.0)
        below = additive._delta_of(s * (1 - 1e-6), eps, 2.0)
        ok &= abs(at - 1e-5) < 1e-9 and below > 1e-5
        rows.append(f'eps {eps:g}: sigma {s:.3f} r')
    check('Gaussian: analytic sigma meets the condition with equality and is the smallest', ok, ', '.join(rows))
    eps, r = 1.0, 0.03
    b = additive.noise_scale('laplace', eps, 0, r)
    # Laplace density ratio between the two interval ends at any output y: exp(|x1 - x2| / b)
    check('Laplace: b = 2r/eps, worst density ratio = e^eps', abs(np.exp(2 * r / b) - np.exp(eps)) < 1e-12)


def test_unbiased_clipping_layers():
    rng = np.random.default_rng(0)
    g = [rng.normal(0, 0.1, 300), rng.normal(0, 0.1, 300)]
    radii = [0.03, 0.01]
    d0 = rng.uniform(-0.03, 0.03, 300)
    d1 = np.where(np.arange(300) % 2 == 0, 0.05, -0.05)        # all beyond r = 0.01
    w = [g[0] + d0, g[1] + d1]
    target = [g[0] + d0, g[1] + np.clip(d1, -0.01, 0.01)]
    for method, var in (('laplace', lambda s: 2 * s * s), ('gaussian', lambda s: s * s)):
        outs = np.stack([np.concatenate(additive.perturb(w, g, radii, 1.0, 1e-5, method, k)[0])
                         for k in range(4000)])
        scales = [additive.noise_scale(method, 1.0, 1e-5, r) for r in radii]
        mean_ok = all(np.max(np.abs(outs[:, i * 300:(i + 1) * 300].mean(0) - target[i]))
                      < 5 * np.sqrt(var(scales[i]) / 4000) for i in range(2))
        var_ok = all(abs(np.var(outs[:, i * 300:(i + 1) * 300] - target[i]) / var(scales[i]) - 1) < 0.05
                     for i in range(2))
        _, clipped = additive.perturb(w, g, radii, 1.0, 1e-5, method, 1)
        check(f'{method}: unbiased for the clipped weight, noise variance as calibrated, per-layer scale',
              mean_ok and var_ok and clipped == [0.0, 1.0] and abs(scales[0] / scales[1] - 3) < 1e-6,
              f'scales {[round(s, 4) for s in scales]}, clipped share {clipped}')


if __name__ == '__main__':
    test_calibration()
    test_unbiased_clipping_layers()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
