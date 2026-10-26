# group_setup.py
# Group-secret setup for the server-hidden CPQ pairing (paper, Appendix "Server-Hidden Pairing",
# Algorithm "Pairing setup and shared-randomness generation"; privacy.cpq_group_key: ml_kem).
#
#   KEY       designated vehicle D (first in the agreed list R) -> every other vehicle u:
#             ML-KEM-512 ciphertext to u's KEM key + K_G encrypted with ChaCha20-Poly1305 under a key
#             derived (HKDF) from the KEM shared secret; signed by D (ML-DSA).
#   CONFIRM   u -> D: key-confirmation tag HMAC(K_check, u | e | H(R)); signed by u.
#   SETUP_OK  D -> every vehicle: setup complete for (e, H(R)); signed by D.
# Every vehicle verifies everything it receives; any failure raises SetupError (abort).
# The server only relays the messages (step 4 sends them over ns-3 via the RSU); it never holds a
# KEM secret key, so it cannot recover K_G.
#
# Keys: each vehicle has a registered ML-DSA signing key pair (pqc_keys/<veh>_<scheme>.pk/.sk, as
# for the model updates) and an ML-KEM-512 key pair (pqc_keys/<veh>_ML-KEM-512.pk/.sk).
# K_G is given by the caller (seeded from the run seed for reproducible simulations); the KEM and
# signature randomness is fresh, which does not change the derived pairing.

import hashlib
import hmac
import json
import os
import struct
import time

import oqs
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

import cpq_pairing
from pqc.pqc_crypto import load_or_create_keypair

KEM = 'ML-KEM-512'
MAGIC = b'CPQS'


class SetupError(Exception):
    """Group setup failed (bad signature, decryption, tag, or protocol mismatch): abort."""


def load_or_create_kem_keypair(veh, key_dir, kem=KEM):
    os.makedirs(key_dir, exist_ok=True)
    pk_path = os.path.join(key_dir, f'{veh}_{kem}.pk')
    sk_path = os.path.join(key_dir, f'{veh}_{kem}.sk')
    if os.path.exists(pk_path) and os.path.exists(sk_path):
        return open(pk_path, 'rb').read(), open(sk_path, 'rb').read()
    with oqs.KeyEncapsulation(kem) as k:
        pk = k.generate_keypair()
        sk = k.export_secret_key()
    with open(pk_path, 'wb') as f:
        f.write(pk)
    with open(sk_path, 'wb') as f:
        f.write(sk)
    return pk, sk


# ── message format: MAGIC | 1-byte type | header json | blobs | signature, length-prefixed ──
TYPES = {'KEY': 1, 'CONFIRM': 2, 'SETUP_OK': 3}


def _pack(kind, header, blobs, sign_sk, scheme):
    body = MAGIC + bytes([TYPES[kind]])
    for part in [json.dumps(header, sort_keys=True).encode()] + list(blobs):
        body += struct.pack('<I', len(part)) + part
    with oqs.Signature(scheme, secret_key=sign_sk) as s:
        sig = s.sign(body)
    return body + struct.pack('<I', len(sig)) + sig


def _unpack(msg, expect_kind):
    if len(msg) < 5 or msg[:4] != MAGIC or msg[4] != TYPES[expect_kind]:
        raise SetupError(f'not a {expect_kind} message')
    try:                                    # any corruption -> abort, never a crash
        parts, k = [], 5
        while k < len(msg):
            (n,) = struct.unpack('<I', msg[k:k + 4])
            parts.append(msg[k + 4:k + 4 + n])
            k += 4 + n
        if k != len(msg) or len(parts) < 2:
            raise ValueError('lengths')
        body = msg[:len(msg) - 4 - len(parts[-1])]
        header = json.loads(parts[0])
        if not isinstance(header, dict):
            raise ValueError('header')
    except Exception:
        raise SetupError(f'malformed {expect_kind} message')
    return header, parts[1:-1], parts[-1], body


def _verify(body, sig, pk, scheme, what):
    with oqs.Signature(scheme) as s:
        if not s.verify(body, sig, pk):
            raise SetupError(f'bad signature on {what}')


def _wrap_key(shared_secret, epoch, roster_hash, recipient):
    info = b'cpq-kg-wrap|' + str(epoch).encode() + b'|' + roster_hash + b'|' + recipient.encode()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(shared_secret)


def _tag(k_check, veh, epoch, roster_hash):
    return hmac.new(k_check, b'confirm|' + veh.encode() + b'|' + str(epoch).encode() + b'|' + roster_hash,
                    hashlib.sha256).digest()


class Vehicle:
    """One vehicle's side of the setup (its own secret keys only)."""

    def __init__(self, veh, roster, scheme, key_dir, epoch=cpq_pairing.EPOCH):
        self.veh, self.roster, self.scheme, self.epoch = veh, list(roster), scheme, epoch
        self.rh = cpq_pairing.roster_hash(self.roster)
        self.sig_pk, self.sig_sk = load_or_create_keypair(veh, scheme, key_dir)
        self.kem_pk, self.kem_sk = load_or_create_kem_keypair(veh, key_dir)
        self.key_dir = key_dir
        self.keys = None            # (K_pair, K_coin, K_check) once K_G is known
        self.confirmed = False      # SETUP_OK received and verified

    def _pub(self, veh):            # registered public keys of other vehicles
        return (load_or_create_keypair(veh, self.scheme, self.key_dir)[0],
                load_or_create_kem_keypair(veh, self.key_dir)[0])

    # designated vehicle
    def key_messages(self, group_secret):
        """KEY message for every other vehicle (designated vehicle only)."""
        if self.veh != self.roster[0]:
            raise SetupError('only the designated vehicle (first in the list) distributes K_G')
        self.keys = cpq_pairing.group_keys(group_secret, self.roster, self.epoch)
        out = {}
        for u in self.roster[1:]:
            with oqs.KeyEncapsulation(KEM) as k:
                ct, ss = k.encap_secret(self._pub(u)[1])
            wrapped = ChaCha20Poly1305(_wrap_key(ss, self.epoch, self.rh, u)).encrypt(
                b'\0' * 12, group_secret, self.rh + u.encode())
            header = {'from': self.veh, 'to': u, 'epoch': self.epoch, 'roster': self.rh.hex(), 'kem': KEM}
            out[u] = _pack('KEY', header, [ct, wrapped], self.sig_sk, self.scheme)
        return out

    def check_confirmations(self, messages):
        """Verify every CONFIRM (designated vehicle); returns the SETUP_OK message."""
        missing = set(self.roster[1:]) - set(messages)
        if missing:
            raise SetupError(f'no confirmation from {sorted(missing)}')
        for u, msg in messages.items():
            header, blobs, sig, body = _unpack(msg, 'CONFIRM')
            _verify(body, sig, self._pub(u)[0], self.scheme, f'CONFIRM from {u}')
            if header.get('from') != u or header.get('epoch') != self.epoch or header.get('roster') != self.rh.hex():
                raise SetupError(f'CONFIRM from {u}: wrong sender, epoch or roster')
            if not hmac.compare_digest(blobs[0], _tag(self.keys[2], u, self.epoch, self.rh)):
                raise SetupError(f'CONFIRM from {u}: wrong key-confirmation tag')
        self.confirmed = True
        header = {'from': self.veh, 'epoch': self.epoch, 'roster': self.rh.hex(), 'status': 'ok'}
        return _pack('SETUP_OK', header, [], self.sig_sk, self.scheme)

    # every other vehicle
    def receive_key(self, msg):
        """Verify the KEY message, recover K_G, derive the keys; returns this vehicle's CONFIRM."""
        header, blobs, sig, body = _unpack(msg, 'KEY')
        designated = self.roster[0]
        _verify(body, sig, self._pub(designated)[0], self.scheme, 'KEY')
        if (header.get('from') != designated or header.get('to') != self.veh
                or header.get('epoch') != self.epoch or header.get('roster') != self.rh.hex()):
            raise SetupError('KEY: wrong sender, recipient, epoch or roster')
        ct, wrapped = blobs
        with oqs.KeyEncapsulation(KEM, secret_key=self.kem_sk) as k:
            ss = k.decap_secret(ct)
        try:
            group_secret = ChaCha20Poly1305(_wrap_key(ss, self.epoch, self.rh, self.veh)).decrypt(
                b'\0' * 12, wrapped, self.rh + self.veh.encode())
        except Exception:
            raise SetupError('KEY: K_G does not decrypt (wrong ciphertext or key)')
        self.keys = cpq_pairing.group_keys(group_secret, self.roster, self.epoch)
        header = {'from': self.veh, 'to': designated, 'epoch': self.epoch, 'roster': self.rh.hex()}
        return _pack('CONFIRM', header, [_tag(self.keys[2], self.veh, self.epoch, self.rh)], self.sig_sk, self.scheme)

    def receive_setup_ok(self, msg):
        header, _, sig, body = _unpack(msg, 'SETUP_OK')
        _verify(body, sig, self._pub(self.roster[0])[0], self.scheme, 'SETUP_OK')
        if header.get('epoch') != self.epoch or header.get('roster') != self.rh.hex() or header.get('status') != 'ok':
            raise SetupError('SETUP_OK: wrong epoch, roster or status')
        self.confirmed = True


def run_in_process(roster, scheme, key_dir, group_secret, log=None):
    """Whole setup with direct message passing (the server relays; nothing is lost).

    Returns {veh: (K_pair, K_coin, K_check)} and per-message records (type, from, to, bytes).
    """
    t0 = time.time()
    vehicles = {v: Vehicle(v, roster, scheme, key_dir) for v in roster}
    D = vehicles[roster[0]]
    records = []
    keys_msgs = D.key_messages(group_secret)
    confirms = {}
    for u, msg in keys_msgs.items():
        records.append(('KEY', D.veh, u, len(msg)))
        confirms[u] = vehicles[u].receive_key(msg)
        records.append(('CONFIRM', u, D.veh, len(confirms[u])))
    ok = D.check_confirmations(confirms)
    for u in roster[1:]:
        vehicles[u].receive_setup_ok(ok)
        records.append(('SETUP_OK', D.veh, u, len(ok)))
    if log:
        log(f'[SETUP] group of {len(roster)} vehicles: {len(records)} messages, '
            f'{sum(r[3] for r in records) / 1000:.1f} KB, {time.time() - t0:.2f} s')
    return {v: veh.keys for v, veh in vehicles.items()}, records
