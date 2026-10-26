# Q-network: 94 state inputs, hidden layers 150 and 100, one output per turn.
import tensorflow as tf
# from tensorflow.python.keras.layers import Dense

# from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform

class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(150, activation= 'relu')
        self.fc2 = Dense(100, activation= 'relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3,1e-3))
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        qvalue = self.fc_out(x)
        return qvalue
    
# Shared layer name list — order must match model.get_weights() output.
# Used by both client DP classes and server aggregation to name layers.
LAYER_NAMES = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc_out.weight', 'fc_out.bias']