"""Builds attack/probe_states.npy: 60 fixed states (30 edges x empty / busy traffic) for the 'probe'
attack features. Run once from the repository root: python attack/make_probe_states.py
"""
import numpy as np
import sumolib

from frl.utils import get_alledges

NET, DEST = 'sumo/net/dqn.net.xml', 'E15'

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
