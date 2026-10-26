# tests/check_dqn.py
# Checks for the DQN target options (dqnagent.py):
#   python tests/check_dqn.py        (inside sumo_env, from the repo root)
#
# 1. mask_target   target = r + gamma * max over the valid next turns only; with the flag off the
#                  max runs over all outputs (earlier behaviour). Network outputs are fixed through
#                  the output-layer bias, so the expected loss is known exactly.
# 2. target copy   the target network is copied every target_update_every gradient steps.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np

from frl.learning.dqnagent import dqnAgent

S, A, GAMMA = 4, 3, 0.99
failures = []


def check(name, ok, detail=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f' — {detail}' if detail else ''), flush=True)
    if not ok:
        failures.append(name)


def fixed_outputs(net, values):
    """Make the network output `values` for every state (zero output kernel, bias = values)."""
    w = net.get_weights()
    w[-2] = np.zeros_like(w[-2])
    w[-1] = np.asarray(values, np.float32)
    net.set_weights(w)


def agent(mask_target, target_update_every=1000):
    a = dqnAgent('veh0', 0, [], {}, S, A, 1, '/tmp', batch_size=1, train_start=1,
                 target_update_every=target_update_every, mask_target=mask_target)
    return a


def huber(x):
    x = abs(x)
    return 0.5 * x * x if x <= 1 else x - 0.5


def loss_for(mask_target, next_valid, done=0.0, target_q=(-62.0, -20.0, -150.0), reward=-20.0):
    a = agent(mask_target)
    fixed_outputs(a.model, [0.0, 0.0, 0.0])            # Q(s, a) = 0 for the chosen action 0
    fixed_outputs(a.target_model, list(target_q))
    a.append_sample(np.zeros(S), 0, reward, np.zeros(S), done, next_valid=next_valid)
    return a.train_model()


def main():
    # E11 example: one real exit (action 0); outputs 1 and 2 are turns that do not exist
    r = -20.0
    unmasked = huber(r + GAMMA * -20.0)                 # max over all: -20 (non-existent turn)
    masked = huber(r + GAMMA * -62.0)                   # max over the real exit: -62
    got_off = loss_for(False, [0])
    got_on = loss_for(True, [0])
    check('flag off: max over all outputs (earlier behaviour)', np.isclose(got_off, unmasked, atol=1e-3),
          f'loss {got_off:.3f}, expected {unmasked:.3f} (target -39.8)')
    check('flag on: max over the valid next turns only', np.isclose(got_on, masked, atol=1e-3),
          f'loss {got_on:.3f}, expected {masked:.3f} (target -81.4)')
    # all turns valid -> identical with and without the mask
    check('all turns valid: same target with and without the mask',
          np.isclose(loss_for(True, [0, 1, 2]), loss_for(False, [0, 1, 2]), atol=1e-6))
    # non-existent turns lower than the real one -> same target
    low = (-62.0, -93.0, -150.0)
    check('non-existent turns below the real one: same target',
          np.isclose(loss_for(True, [0], target_q=low), loss_for(False, [0], target_q=low), atol=1e-6))
    # destination (no exits, terminal): no bootstrap either way
    check('terminal next state with no exits: target = reward',
          np.isclose(loss_for(True, [], done=1.0), huber(r), atol=1e-3))
    # next_valid not given -> all actions (old callers)
    check('next_valid not given: all actions', np.isclose(loss_for(True, None), unmasked, atol=1e-3))

    # target copy period
    a = agent(False, target_update_every=3)
    for _ in range(5):
        a.append_sample(np.random.rand(S), 1, -1.0, np.random.rand(S), 0.0)
    a.train_model(); a.train_model()
    same_before = all(np.allclose(x, y) for x, y in zip(a.model.get_weights(), a.target_model.get_weights()))
    a.train_model()                                     # 3rd step -> copy
    same_after = all(np.allclose(x, y) for x, y in zip(a.model.get_weights(), a.target_model.get_weights()))
    check('target copied every target_update_every steps', (not same_before) and same_after,
          f'after 2 steps equal={same_before}, after 3 equal={same_after}')


if __name__ == '__main__':
    main()
    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)
