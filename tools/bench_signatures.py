"""Signature microbenchmark (Table H.1): public-key and signature sizes and key generation, signing and
verification times (mean ± SD in ms over 200 trials, 4,096-byte messages). Timings depend on the machine.

    python tools/bench_signatures.py [--out results/paper/signature_bench.csv]
"""
import argparse
import os
import time

import numpy as np
import oqs
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

N = 200
MSG = np.random.default_rng(0).bytes(4096)
PQC = [('ML-DSA-44', 'ML-DSA-44', 'MLWE/MSIS'), ('Falcon-512', 'Falcon-512', 'NTRU/SIS'),
       ('SLH-DSA-SHA2-128f', 'SLH_DSA_PURE_SHA2_128F', 'Hash'), ('SLH-DSA-SHA2-128s', 'SLH_DSA_PURE_SHA2_128S', 'Hash')]


def timed(f):
    a = time.perf_counter()
    out = f()
    return out, 1e3 * (time.perf_counter() - a)


def bench_pqc(name):
    kg, sg, vf, pk_len, sig_len = [], [], [], 0, []
    for _ in range(N):
        with oqs.Signature(name) as s:
            pk, t = timed(s.generate_keypair)
            kg.append(t)
            sig, t = timed(lambda: s.sign(MSG))
            sg.append(t)
            ok, t = timed(lambda: s.verify(MSG, sig, pk))
            assert ok
            vf.append(t)
            pk_len, sig_len = len(pk), sig_len + [len(sig)]
    return pk_len, float(np.mean(sig_len)), kg, sg, vf


def bench_classical(kind):
    kg, sg, vf, sig_len = [], [], [], []
    for _ in range(N):
        if kind == 'rsa':
            sk, t = timed(lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048))
            pss = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
            sig, ts = timed(lambda: sk.sign(MSG, pss, hashes.SHA256()))
            _, tv = timed(lambda: sk.public_key().verify(sig, MSG, pss, hashes.SHA256()))
            pk_len = sk.public_key().public_numbers().n.bit_length() // 8
        else:
            sk, t = timed(lambda: ec.generate_private_key(ec.SECP256R1()))
            sig, ts = timed(lambda: sk.sign(MSG, ec.ECDSA(hashes.SHA256())))
            _, tv = timed(lambda: sk.public_key().verify(sig, MSG, ec.ECDSA(hashes.SHA256())))
            pk_len = len(sk.public_key().public_bytes(serialization.Encoding.X962,
                                                      serialization.PublicFormat.UncompressedPoint))
        kg.append(t); sg.append(ts); vf.append(tv); sig_len.append(len(sig))
    return pk_len, float(np.mean(sig_len)), kg, sg, vf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=os.path.join('results', 'paper', 'signature_bench.csv'))
    args = p.parse_args()
    rows = []
    for label, name, basis in PQC:
        rows.append((label, basis) + bench_pqc(name))
    rows.append(('RSA-2048-PSS-SHA256', 'IFP') + bench_classical('rsa'))
    rows.append(('ECDSA-P256-SHA256', 'ECDLP') + bench_classical('ecdsa'))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('algorithm,basis,pk_bytes,sig_bytes_mean,keygen_ms_mean,keygen_ms_sd,sign_ms_mean,sign_ms_sd,'
                'verify_ms_mean,verify_ms_sd\n')
        print(f'{"Algorithm":22s} {"Basis":10s} {"PK (B)":>7s} {"Sig. (B)":>9s} {"Keygen (ms)":>16s} '
              f'{"Sign (ms)":>18s} {"Verify (ms)":>16s}')
        for label, basis, pk, sig, kg, sg, vf in rows:
            stats = [(np.mean(x), np.std(x, ddof=1)) for x in (kg, sg, vf)]
            f.write(f'{label},{basis},{pk},{sig:.1f},' + ','.join(f'{m:.4f},{s:.4f}' for m, s in stats) + '\n')
            print(f'{label:22s} {basis:10s} {pk:7d} {sig:9.1f} ' +
                  ' '.join(f'{m:8.3f} ± {s:6.3f}' for m, s in stats))
    print('->', args.out)


if __name__ == '__main__':
    main()
