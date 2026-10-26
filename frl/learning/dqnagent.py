# dqnagent.py
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from frl.learning.dqn import DQN


class dqnAgent:
    def __init__(
        self,
        id,
        num_seed,
        edgelists,
        dict_connection,
        state_size,
        action_size,
        num_episode,
        dirModel,
        *,
        discount_factor=0.99,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_decay=1.0,
        epsilon_min=0.05,
        batch_size=32,
        train_start=200,
        target_update_every=1000,   # steps between hard target copies
        huber_delta=1.0,
        mask_target=False,          # target max over the valid turns of the next edge only
    ):
        # meta
        self.id = id
        self.edgelists = edgelists
        self.dict_connection = dict_connection
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.num_episode = num_episode
        self.dirModel = dirModel
        self.num_seed = num_seed
        # Own RNG: exploration and replay sampling must not depend on the global
        # np.random state (shared with other code that may reseed it).
        self.rng = np.random.default_rng(num_seed)

        # DQN hyperparams
        self.discount_factor = float(discount_factor)
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon_start)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.batch_size = int(batch_size)
        self.train_start = int(train_start)
        self.target_update_every = int(target_update_every)
        self.mask_target = bool(mask_target)
        self.huber = tf.keras.losses.Huber(delta=float(huber_delta), reduction="none")

        # replay buffer
        self.memory = deque(maxlen=20_000)

        # networks & optimizer
        self.model = DQN(self.action_size)
        self.target_model = DQN(self.action_size)
        self.optimizer = Adam(learning_rate=self.learning_rate)

        # counters
        self.global_steps = 0

        # build once to initialize weights/vars and avoid None grads
        self._build_models()
        self.update_target_model()

    # ---------- utils ----------

    def _build_models(self):
        """Build/warm up models so variables exist before train/save."""
        dummy = tf.zeros((1, self.state_size), dtype=tf.float32)
        _ = self.model(dummy, training=False)
        _ = self.target_model(dummy, training=False)

    @staticmethod
    def _ensure_2d_state(state):
        """Ensures np.float32 array of shape (1, state_size)."""
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        return state

    # ---------- public API ----------

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def maybe_update_target(self):
        if self.target_update_every > 0 and (self.global_steps % self.target_update_every == 0):
            self.update_target_model()

    def get_action(self, state, valid_actions=None):
        state = self._ensure_2d_state(state)
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        if self.rng.random() <= self.epsilon:
            return int(self.rng.choice(valid_actions))
        q = self.model(state, training=False).numpy()[0]
        masked = np.full(self.action_size, -np.inf)
        masked[valid_actions] = q[valid_actions]
        return int(np.argmax(masked))

    def append_sample(self, state, action, reward, next_state, done, next_valid=None):
        """Store transitions as flat arrays; done stored as 0.0/1.0.
        next_valid: valid actions at the next state (for mask_target), stored as a bit mask;
        None or empty = all actions."""
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
        a = int(action)
        r = float(reward)
        d = float(bool(done))
        bits = sum(1 << int(i) for i in next_valid) if next_valid else (1 << self.action_size) - 1
        self.memory.append((s, a, r, ns, d, bits))

    def train_model(self):
        """One gradient step; safe to call every env step."""
        if len(self.memory) < max(self.train_start, self.batch_size):
            return None  # not enough data yet

        idx = self.rng.choice(len(self.memory), size=self.batch_size, replace=False)
        batch = [self.memory[i] for i in idx]

        states = np.stack([b[0] for b in batch]).astype(np.float32)      # (B, S)
        actions = np.array([b[1] for b in batch], dtype=np.int32)        # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)      # (B,)
        next_states = np.stack([b[3] for b in batch]).astype(np.float32) # (B, S)
        dones = np.array([b[4] for b in batch], dtype=np.float32)        # (B,)
        bits = np.array([b[5] for b in batch], dtype=np.int64)
        next_valid = ((bits[:, None] >> np.arange(self.action_size)) & 1).astype(np.float32)  # (B, A)

        loss = self._train_step(
            tf.convert_to_tensor(states), tf.convert_to_tensor(actions),
            tf.convert_to_tensor(rewards), tf.convert_to_tensor(next_states),
            tf.convert_to_tensor(dones), tf.convert_to_tensor(next_valid),
        )
        self.global_steps += 1
        self.maybe_update_target()
        # return scalar loss for logging (as python float)
        return float(loss.numpy())

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, next_valid):
        with tf.GradientTape() as tape:
            # Q(s, ·)
            q_pred = self.model(states, training=True)                   # (B, A)
            idx = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            q_sa = tf.gather_nd(q_pred, idx)                             # (B,)

            # Target: r + γ * max_a' Q_target(s', a') * (1 - done)
            q_next = self.target_model(next_states, training=False)      # (B, A)
            if self.mask_target:
                # max over the turns that exist at the next edge (none, e.g. destination:
                # all outputs, only reached with done = 1)
                none_valid = tf.reduce_sum(next_valid, axis=1, keepdims=True) == 0
                valid = tf.logical_or(next_valid > 0, none_valid)
                q_next = tf.where(valid, q_next, tf.fill(tf.shape(q_next), -1e9))
            max_q_next = tf.reduce_max(q_next, axis=1)                   # (B,)
            targets = rewards + (1.0 - dones) * self.discount_factor * max_q_next

            # Huber loss (per-sample then mean)
            loss = tf.reduce_mean(self.huber(targets, q_sa))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # ---------- weights I/O ----------

    # dqnagent.py
    # dqnagent.py

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
                
    def save_weights(self):
        os.makedirs(self.dirModel, exist_ok=True)
        # OLD:
        # path = os.path.join(self.dirModel, f"{self.num_episode}dqn{self.id}.h5")
        # NEW:
        path = os.path.join(self.dirModel, f"{self.num_episode}dqn{self.id}.weights.h5")
        self._build_models()
        self.model.save_weights(path)

    # def save_weights(self, name=None):
    #     import os
    #     os.makedirs(self.dirModel, exist_ok=True)
    #     self._build_models()
    #     # ensure correct suffix
    #     fname = name or f"{self.num_episode}dqn{self.id}.weights.h5"
    #     if not fname.endswith(".weights.h5"):
    #         fname = fname.rsplit(".", 1)[0] + ".weights.h5"
    #     path = os.path.join(self.dirModel, fname)
    #     self.model.save_weights(path)
    #     return path

    def load_weights(self, path):
        self._build_models()
        # accept old ".h5" and new ".weights.h5"
        if path.endswith(".h5") and not path.endswith(".weights.h5"):
            path = path.rsplit(".h5", 1)[0] + ".weights.h5"
        if os.path.isfile(path):
            self.model.load_weights(path)
            self.update_target_model()
            return True
        return False

    # ---------- optional helpers for FL clients ----------

    def get_parameters_numpy(self):
        self._build_models()
        return [w.numpy() for w in self.model.get_weights()]

    def set_parameters_numpy(self, params_list):
        self._build_models()
        self.model.set_weights(params_list)
        self.update_target_model()
