"""Probe states for the function-space features (attack/features.py, feature set 'probe'). Run once from the repo
root in conda env sumo_env (needs sumolib): python attack/make_probe_states.py -> attack/probe_states.npy

The state layout is that of shared_env.get_state_np ('raw' encoding, 94 inputs): vehicle count per edge (30),
mean speed per edge (30), length of lane 0 per edge (30), start of lane 0 of the current edge (x, y), start of
lane 0 of the destination E15 (x, y). Edge order = utils.get_alledges(Net/dqn.net.xml).

One probe state per (edge the vehicle is on, traffic level), 30 edges x 2 levels = 60 states:
  empty     0 vehicles, mean speed = speed limit (what SUMO reports for an empty edge)
  busy      a third of the lane filled (length / 7.5 m * 0.33 vehicles), half the speed limit
The attacker only needs states that mean the same in every run; it knows the map and the state code.
"""
import numpy as np
import sumolib

from utils import get_alledges

NET, DEST = 'Net/dqn.net.xml', 'E15'

edges = get_alledges(NET)
net = sumolib.net.readNet(NET)
lanes = [net.getLane(e + '_0') for e in edges]
length = np.array([l.getLength() for l in lanes])
limit = np.array([l.getSpeed() for l in lanes])
dest = net.getLane(DEST + '_0').getShape()[0]
states, info = [], []
for level, counts, speeds in (('empty', np.zeros(len(edges)), limit),
                              ('busy', np.floor(length / 7.5 * 0.33), 0.5 * limit)):
    for e, lane in zip(edges, lanes):
        start = lane.getShape()[0]
        states.append(np.concatenate([counts, speeds, length, start, dest]))
        info.append(f'{level}:{e}')
states = np.asarray(states, np.float32)
assert states.shape == (60, 94), states.shape
np.save('attack/probe_states.npy', states)
with open('attack/probe_states.txt', 'w') as f:
    f.write('\n'.join(info) + '\n')
print('attack/probe_states.npy', states.shape)
