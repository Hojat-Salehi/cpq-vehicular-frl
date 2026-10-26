# additive.py
# Additive-noise baselines (paper, Sec. "Gaussian and Laplace Mechanisms"), on the same public
# clipping interval as the one-bit methods (onebit.py):
#
#   interval  coordinate j of layer l: [w_g,j - r_l, w_g,j + r_l] (w_g = global model, r_l =
#             privacy.clip_radius), i.e. each weight's update is clipped to [-r_l, r_l];
#             per-coordinate sensitivity Delta_l = 2 r_l.
#   laplace   w + Lap(0, b_l), b_l = 2 r_l / eps                       -> pure eps-PLDP
#   gaussian  w + N(0, sigma_l^2), sigma_l = analytic Gaussian calibration (Balle & Wang 2018):
#             the smallest sigma with
#             Phi(D/(2s) - eps s/D) - e^eps Phi(-D/(2s) - eps s/D) <= delta   -> (eps, delta)-PLDP
#
# Noisy weights are sent as float32; the server averages them (plain FedAvg).

from functools import lru_cache

import numpy as np

METHODS = ('gaussian', 'laplace')


def _delta_of(sigma, eps, sens):
    from scipy.stats import norm
    a = sens / (2.0 * sigma)
    b = eps * sigma / sens
    return norm.cdf(a - b) - np.exp(eps) * norm.cdf(-a - b)


@lru_cache(maxsize=None)
def analytic_sigma(eps, delta, sens):
    """Smallest sigma for which the Gaussian mechanism is (eps, delta)-DP at sensitivity sens."""
    from scipy.optimize import brentq
    lo, hi = 1e-8 * sens, 1e4 * sens          # delta(sigma) decreases in sigma
    return float(brentq(lambda s: _delta_of(s, eps, sens) - delta, lo, hi, xtol=1e-14 * sens, rtol=1e-12))


def noise_scale(method, eps, delta, radius):
    """Laplace scale b or Gaussian sigma for one layer."""
    if method == 'laplace':
        return 2.0 * radius / eps
    if method == 'gaussian':
        return analytic_sigma(float(eps), float(delta), 2.0 * radius)
    raise ValueError(f'unknown additive method {method!r}')


def perturb(weights, global_weights, radii, eps, delta, method, seed):
    """Clip each weight to [w_g - r, w_g + r] and add the layer's noise.

    Returns the noisy weights (float32) and the per-layer share of clipped coordinates.
    """
    if method not in METHODS:
        raise ValueError(f'unknown additive method {method!r}')
    rng = np.random.default_rng(int(seed))
    out, clipped = [], []
    for w, g, r in zip(weights, global_weights, radii):
        w = np.asarray(w, np.float64)
        g = np.asarray(g, np.float64)
        d = w - g
        clipped.append(float(np.mean(np.abs(d) > r)))
        scale = noise_scale(method, eps, delta, r)
        noise = rng.laplace(0.0, scale, w.shape) if method == 'laplace' else rng.normal(0.0, scale, w.shape)
        out.append((g + np.clip(d, -r, r) + noise).astype(np.float32))
    return out, clipped
