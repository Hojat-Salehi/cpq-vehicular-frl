# tests/check_group_setup.py
# Checks for the ML-KEM group setup of the server-hidden CPQ pairing (group_setup.py).
#   python tests/check_group_setup.py        (inside sumo_env, from the repo root)
#
# 1. keys        a full setup gives every vehicle the same keys as cpq_pairing.group_keys(K_G)
# 2. sizes       message sizes (KEY, CONFIRM, SETUP_OK)
# 3. tampering   modified ciphertext, forged sender, KEY sent to the wrong vehicle, wrong K_G
#                (wrong confirmation tag), missing confirmation, modified SETUP_OK -> abort
# 4. secrecy     a KEY message cannot be opened with another vehicle's KEM key (the relaying
#                server holds none)

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from frl.privacy import cpq_pairing
from frl.privacy import group_setup
from frl.privacy.group_setup import SetupError, Vehicle

SCHEME = 'ML-DSA-44'
failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def aborts(fn):
    try:
        fn()
    except SetupError as e:
        return str(e)
    return None


def main():
    key_dir = tempfile.mkdtemp()
    roster = [f'veh{k}' for k in range(6)]
    K = cpq_pairing.seeded_group_secret(5)
    keys, records = group_setup.run_in_process(roster, SCHEME, key_dir, K)
    expected = cpq_pairing.group_keys(K, roster)
    check('full setup: every vehicle holds the keys derived from K_G', all(keys[v] == expected for v in roster))
    size = {t: [b for kind, _, _, b in records if kind == t] for t in ('KEY', 'CONFIRM', 'SETUP_OK')}
    check('message sizes', len(size['KEY']) == 5 and len(size['CONFIRM']) == 5 and len(size['SETUP_OK']) == 5,
          ', '.join(f'{t} {v[0]} B x {len(v)}' for t, v in size.items())
          + f'; total {sum(r[3] for r in records)} B for {len(roster)} vehicles')

    D = Vehicle(roster[0], roster, SCHEME, key_dir)
    msgs = D.key_messages(K)
    u1, u2 = Vehicle('veh1', roster, SCHEME, key_dir), Vehicle('veh2', roster, SCHEME, key_dir)
    bad = bytearray(msgs['veh1'])
    bad[200] ^= 0xFF                                                     # inside the KEM ciphertext
    results = {'modified KEY message': aborts(lambda: u1.receive_key(bytes(bad))),
               'KEY delivered to the wrong vehicle': aborts(lambda: u2.receive_key(msgs['veh1']))}
    forger = Vehicle('veh3', roster, SCHEME, key_dir)
    forger.roster = roster                                               # veh3 pretends to be veh0
    forged = group_setup._pack('KEY', {'from': 'veh0', 'to': 'veh1', 'epoch': 1, 'roster': forger.rh.hex(),
                                       'kem': 'ML-KEM-512'}, [b'x' * 768, b'y' * 48], forger.sig_sk, SCHEME)
    results['KEY forged by another vehicle'] = aborts(lambda: u1.receive_key(forged))
    confirms = {u: Vehicle(u, roster, SCHEME, key_dir).receive_key(msgs[u]) for u in roster[1:]}
    other = Vehicle('veh4', roster, SCHEME, key_dir)
    other.keys = cpq_pairing.group_keys(cpq_pairing.seeded_group_secret(999), roster)
    wrong = dict(confirms)
    wrong['veh4'] = group_setup._pack('CONFIRM', {'from': 'veh4', 'to': 'veh0', 'epoch': 1, 'roster': other.rh.hex()},
                                      [group_setup._tag(other.keys[2], 'veh4', 1, other.rh)], other.sig_sk, SCHEME)
    results['confirmation tag from a wrong K_G'] = aborts(lambda: D.check_confirmations(wrong))
    results['missing confirmation'] = aborts(lambda: D.check_confirmations({u: m for u, m in confirms.items() if u != 'veh5'}))
    ok = D.check_confirmations(confirms)
    bad_ok = bytearray(ok)
    bad_ok[10] ^= 0x01
    results['modified SETUP_OK'] = aborts(lambda: Vehicle('veh1', roster, SCHEME, key_dir).receive_setup_ok(bytes(bad_ok)))
    check('tampering aborts the setup', all(results.values()),
          '; '.join(f'{k}: {v or "NOT detected"}' for k, v in results.items()))

    # the relaying server: another vehicle's KEM secret key cannot open veh1's KEY message
    spy = Vehicle('veh2', roster, SCHEME, key_dir)
    spy.veh = 'veh1'                                                     # use veh2's KEM key on veh1's message
    check('secrecy: a KEY message cannot be opened with another KEM key', aborts(lambda: spy.receive_key(msgs['veh1'])) is not None)
    shutil.rmtree(key_dir)


if __name__ == '__main__':
    main()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
