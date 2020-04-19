from .util import Memory
import numpy as np


class BaseModel:
    def __init__(self, action_n, state_n, env_shape, learning_rate=0.01, reward_decay=0.9, epsilon=0.5,
                 memory_capacity=20000, batch_size=32, update_freq=100, prioritized=False):

        self.action_n = action_n
        self.state_n = state_n
        self.lr = learning_rate
        self.gamma = reward_decay
        self.env_shape = env_shape
        self.epsilon = epsilon
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.update_freq = update_freq

        self.step_counter = 0
        self.memory_counter = 0
        self.backward_count = 0

        self.prioritized = prioritized

        if self.prioritized:
            self.memory = Memory(self.memory_capacity)
        else:
            self.memory = np.zeros((self.memory_capacity, (
                        self.state_n[0][2] * self.state_n[0][0] * self.state_n[0][1] + self.state_n[1]) * 2 + 2))

    def store_transition(self, s, a, r, s_):
        s = np.append(np.reshape(s[0], -1), s[1])  # [(4, 160, 380), 28] -> 4 * 160 * 380 + 28
        s_ = np.append(np.reshape(s_[0], -1), s_[1])
        transition = np.hstack((s, [a, r], s_))
        if self.prioritized:
            self.memory.store(transition)
        else:
            # replace the old memory with new memory
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        raise NotImplementedError

    def choose_action(self, x):
        raise NotImplementedError
