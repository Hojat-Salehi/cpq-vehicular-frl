# onebit.py
# One-bit private quantization of model updates: CPQ (correlated, paired) and LDP-FL
# (independent), following the paper's quantization rule (Sec. "Correlated Private
# Quantizer (CPQ)") with these choices for the public clipping interval and randomness:
#
#   interval  coordinate j of layer l: [c_j - r_l, c_j + r_l] with c_j = w_g,j (the global model
#             the client started from, already known to every client and the server) and r_l a
#             fixed per-layer radius (config privacy.clip_radius). The client's weight is clipped
#             to this interval, i.e. its update w - w_g is clipped to [-r_l, r_l].
#   bit       q = 1/2 + (w - c) / (2 r alpha),  alpha = (e^eps + 1) / (e^eps - 1),
#             b = 1{V < q} with V uniform on [0, 1).
#   CPQ       the two clients of a pair draw the same V per coordinate from their shared pair seed;
#             the leader uses V, the follower 1 - V (negatively correlated errors).
#             The simple shared-seed version: the server assigns pairs and seeds (the paper's
#             hidden pairing / ChaCha20 construction is not implemented).
#   CPQ-lambda  (paper, "CPQ-lambda construction") each client also draws, per coordinate, a
#             private coin C ~ Bernoulli(lambda) and a private uniform R, and uses the threshold
#             T = V (leader) / 1 - V (follower) if C = 1, else R. lambda = 1 is plain CPQ,
#             lambda = 0 is LDP-FL. Every bit keeps P(b = 1) = q, so decoding is unchanged.
#             Against an observer who knows the shared randomness (e.g. a server that assigned
#             the seeds) the privacy is eps_C = ln((e^eps + lambda) / (1 - lambda)).
#   LDP-FL    every client draws its own independent V.
#   decode    w_hat = c + alpha * r * (2b - 1); unbiased for the clipped weight. The server
#             averages the decoded models of the participating clients.
#
# Wire format (unchanged): [packed bits of layer 0, shape of layer 0, packed bits of layer 1, ...].

import numpy as np

METHODS = ('cpq', 'ldpfl')


def alpha(epsilon):
    return (np.exp(epsilon) + 1.0) / (np.exp(epsilon) - 1.0)


def radii_for(weights, clip_radius, layer_names):
    """Per-layer radius list from the config (dict by layer name, or one number for all)."""
    if isinstance(clip_radius, dict):
        missing = [n for n in layer_names if n not in clip_radius]
        if missing:
            raise ValueError(f'privacy.clip_radius has no value for layers {missing}')
        return [float(clip_radius[n]) for n in layer_names[:len(weights)]]
    return [float(clip_radius)] * len(weights)


def bit_probability(weights, global_weights, radii, eps):
    """q per coordinate (list of arrays) for weights clipped to [w_g - r, w_g + r]."""
    a = alpha(eps)
    q = []
    for w, g, r in zip(weights, global_weights, radii):
        d = np.clip(np.asarray(w, np.float64) - np.asarray(g, np.float64), -r, r)
        q.append(0.5 + d / (2.0 * r * a))
    return q


def uniforms(seed, shapes):
    """One uniform value per coordinate of every layer, from one seeded stream."""
    rng = np.random.default_rng(int(seed))
    return [rng.random(s) for s in shapes]


def epsilon_seed_aware(eps, lam):
    """Paper's eps_C: privacy of one client against an observer who knows the shared randomness."""
    return float('inf') if lam >= 1 else float(np.log((np.exp(eps) + lam) / (1.0 - lam)))


def encode(weights, global_weights, radii, eps, method, seed, role=1, cpq_lambda=1.0, private_seed=None,
           shared_V=None):
    """Bits of one client.

    cpq:   seed = the pair seed (shared by both clients of the pair), role 1 = leader (V),
           0 = follower (1 - V).
    ldpfl: seed = a seed of this client alone (independent randomness).
    cpq_lambda < 1 (CPQ-lambda): private_seed = a seed of this client alone, for the private
           coin and uniform that replace the shared value with probability 1 - cpq_lambda.
    shared_V: per-coordinate pairing (cpq_pairing.coordinate_values): the client's shared values
           per layer with its role already applied; seed and role are then not used.
    Returns the packed wire format and the per-layer share of clipped coordinates.
    """
    if method not in METHODS:
        raise ValueError(f'unknown one-bit method {method!r}')
    q = bit_probability(weights, global_weights, radii, eps)
    if method == 'cpq' and shared_V is not None:
        V = [np.asarray(v, np.float64).reshape(x.shape) for v, x in zip(shared_V, q)]
    else:
        V = uniforms(seed, [x.shape for x in q])
        if method == 'cpq' and int(role) == 0:
            V = [1.0 - v for v in V]
    if method == 'cpq' and cpq_lambda < 1.0:
        if private_seed is None:
            raise ValueError('CPQ-lambda needs a private seed')
        prng = np.random.default_rng(int(private_seed))
        V = [np.where(prng.random(v.shape) < cpq_lambda, v, prng.random(v.shape)) for v in V]
    packed, clipped = [], []
    for qi, vi, w, g, r in zip(q, V, weights, global_weights, radii):
        bits = (vi < qi).astype(np.uint8)
        packed += [np.packbits(bits.ravel()), np.array(qi.shape, dtype=np.int32)]
        clipped.append(float(np.mean(np.abs(np.asarray(w, np.float64) - np.asarray(g, np.float64)) > r)))
    return packed, clipped


def decode(packed, global_weights, radii, eps):
    """Model reconstructed from one client's bits: w_g + alpha * r * (2b - 1)."""
    a = alpha(eps)
    out = []
    for k, (g, r) in enumerate(zip(global_weights, radii)):
        bits_packed, shape = packed[2 * k], np.asarray(packed[2 * k + 1]).tolist()
        n = int(np.prod(shape))
        bits = np.unpackbits(np.asarray(bits_packed, np.uint8))[:n].reshape(shape).astype(np.float64)
        out.append((np.asarray(g, np.float64) + a * r * (2.0 * bits - 1.0)).astype(np.float32))
    return out
