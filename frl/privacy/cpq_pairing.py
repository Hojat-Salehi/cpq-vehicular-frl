# Server-hidden per-coordinate pairing for CPQ. From the group keys, the round and the agreed list of
# vehicles, every vehicle derives its partner and the shared uniform V for each coordinate (ChaCha20 as
# PRF); the server cannot reproduce them.
#   privacy.cpq_pairing    local: keys derived from the group secret K_G; server: keys from a seed
#                          sent by the server
#   privacy.cpq_group_key  ml_kem: K_G from the group setup (group_setup.py); seeded: from the run seed

import hashlib
from functools import lru_cache

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

EPOCH = 1   # group configuration identifier e (one fixed group of vehicles per run)


def derive_key(secret, label):
    """32-byte key for one purpose ('pair' or 'coin') from a secret (bytes or int seed)."""
    if isinstance(secret, (int, np.integer)):
        secret = int(secret).to_bytes(16, 'little', signed=False)
    return hashlib.sha256(b'cpq-pairing|' + label.encode() + b'|' + secret).digest()


def roster_hash(roster):
    """H(R): hash of the agreed ordered participant list."""
    return hashlib.sha256('\n'.join(roster).encode()).digest()


def seeded_group_secret(seed):
    """K_G for cpq_group_key: seeded (all vehicles derive the same secret from the run seed;
    simulation only, no setup messages)."""
    return hashlib.sha256(b'cpq-group-secret|' + int(seed).to_bytes(16, 'little')).digest()


def group_keys(group_secret, roster, epoch=EPOCH):
    """(K_pair, K_coin, K_check) = KDF(K_G, e, H(R)), one HKDF-SHA256 call per purpose."""
    out = []
    for label in ('pair', 'coin', 'check'):
        info = b'cpq|' + label.encode() + b'|' + str(epoch).encode() + b'|' + roster_hash(roster)
        out.append(HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(group_secret))
    return tuple(out)


def _stream(key, *index, nbytes):
    nonce = hashlib.sha256(b'|'.join(str(i).encode() for i in index)).digest()[:16]
    return Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor().update(b'\0' * nbytes)


def _uint64(key, index, m):
    return np.frombuffer(_stream(key, *index, nbytes=8 * m), dtype='<u8')


@lru_cache(maxsize=4)
def _round_values(key_pair, key_coin, t, n, m, per_coordinate=True):
    """For every participant (row) and coordinate (column): its V (with its role applied) and
    whether it is paired. Computed once per round and shared by all participants (identical result
    to each participant computing it on its own)."""
    if per_coordinate:
        keys = np.stack([_uint64(key_pair, ('sort', t, u), m) for u in range(n)])    # (n, m)
    else:                                   # one ordering per round, used for every coordinate
        keys = np.repeat(np.stack([_uint64(key_pair, ('sort', t, u), 1) for u in range(n)]), m, axis=1)
    order = np.argsort(keys, axis=0, kind='stable')          # ties broken by list position
    pos = np.empty_like(order)
    pos[order, np.arange(m)[None, :]] = np.arange(n)[:, None]                        # position of u at j
    n_pairs = n // 2
    coin = np.stack([_uint64(key_coin, ('coin', t, l), m) for l in range(max(n_pairs, 1))])
    V_pair = (coin >> np.uint64(11)).astype(np.float64) / float(1 << 53)             # (pairs, m) in [0, 1)
    pair = pos // 2
    paired = pair < n_pairs
    V = np.zeros((n, m))
    cols = np.broadcast_to(np.arange(m), (n, m))
    V[paired] = V_pair[pair[paired], cols[paired]]
    V = np.where((pos % 2 == 1) & paired, 1.0 - V, V)
    V.setflags(write=False)
    pos.setflags(write=False)
    return V, paired, pos


def coordinate_values(pair_secret, t, index, n, shapes, private_seed):
    """Server-seeded per-coordinate pairing (cpq_pairing: server): V for participant `index`
    (0-based position in the list of n participants) in round t, one array per layer shape."""
    return shared_values(derive_key(pair_secret, 'pair'), derive_key(pair_secret, 'coin'),
                         t, index, n, shapes, private_seed, per_coordinate=True)


def shared_values(key_pair, key_coin, t, index, n, shapes, private_seed, per_coordinate=True):
    """V for participant `index` in round t from the pairing keys, one array per layer shape.
    Unpaired coordinates (odd n) get private uniforms from private_seed."""
    m = int(sum(int(np.prod(s)) for s in shapes))
    V, paired, _ = _round_values(key_pair, key_coin, int(t), int(n), m, bool(per_coordinate))
    v = V[index].copy()
    if not paired[index].all():
        own = np.random.default_rng(int(private_seed)).random(m)
        v[~paired[index]] = own[~paired[index]]
    out, k = [], 0
    for s in shapes:
        size = int(np.prod(s))
        out.append(v[k:k + size].reshape(s))
        k += size
    return out


def partners(pair_secret, t, n, m, keys=None, per_coordinate=True):
    """Partner of every participant at every coordinate (-1 = unpaired); for tests.
    keys = (K_pair, K_coin) for local keys, else derived from the server pairing seed."""
    kp, kc = keys if keys is not None else (derive_key(pair_secret, 'pair'), derive_key(pair_secret, 'coin'))
    _, paired, pos = _round_values(kp, kc, int(t), int(n), int(m), bool(per_coordinate))
    order = np.argsort(pos, axis=0)                          # participant at each position
    partner_pos = pos ^ 1
    out = np.where(paired & (partner_pos < n), np.take_along_axis(order, np.minimum(partner_pos, n - 1), 0), -1)
    return out
