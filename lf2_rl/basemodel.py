from .util import Memory, LeafData
import numpy as np


class BaseModel:
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, momentum=0.9, reward_decay=0.9, epsilon=0.5,
                 epsilon_decay=0.9995, memory_capacity=20000, batch_size=32, update_freq=100, save_freq=500,
                 prioritized=False, weight_path=None):
        self.action_n = action_n
        self.state_n = state_n
        self.lr = learning_rate
        self.momentum = momentum
        self.gamma = reward_decay
        self.env_shape = env_shape
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.save_freq = save_freq

        self.step_counter = 0
        self.memory_counter = 0
        self.backward_count = 0

        self.prioritized = prioritized

        if self.prioritized:
            self.memory = Memory(self.memory_capacity)
        else:
            self.memory = np.zeros(self.memory_capacity, dtype=LeafData)
        self.weight_path = weight_path

    @ staticmethod
    def reward_modify(r):
        return r

    def policy(self):
        return self.epsilon

    def store_transition(self, s, a, r, s_, done):
        r = self.reward_modify(r)
        leaf_data = LeafData(s, s_, a, r, done)
        if isinstance(self.memory, Memory):
            self.memory.store(leaf_data)
        else:
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_capacity
            self.memory[index] = leaf_data
        self.memory_counter += 1

    @ staticmethod
    def data_process(data):
        return data.astype('float32') / 255.

    @ staticmethod
    def trans_obser(*args):
        raise NotImplementedError

    def save_weight(self, path):
        raise NotImplementedError

    def restore_weight(self, path):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def choose_action(self, x):
        raise NotImplementedError
