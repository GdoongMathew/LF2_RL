import numpy as np


class LeafData:
    state_0 = None
    state_1 = None
    action = None
    reward = None

    def __init__(self, state_0, state_1, action, reward):
        self.state_0 = state_0
        self.state_1 = state_1
        self.action = action
        self.reward = reward


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while parent_idx * 2 + 1 <= self.capacity:
            l_idx = parent_idx * 2 + 1
            r_idx = parent_idx * 2 + 2
            if self.tree[parent_idx] > v:
                parent_idx = l_idx
            else:
                parent_idx = r_idx
                v -= self.tree[l_idx]

        data_idx = parent_idx + 1 - self.capacity
        return parent_idx, self.tree[parent_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory:

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.tree = SumTree(self.capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper

        self.tree.add(max_p, transition)

    def sample(self, n):
        IS_weight = np.empty((n, 1))
        sample_idx = np.empty((n, ))
        sample_mem = np.empty((n, self.tree.data[0].size))

        segment = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.capacity:]) / n

        for i in range(n):
            l_p, h_p = segment * i, segment * (i + 1)
            p_val = np.random.uniform(l_p, h_p)
            idx, p, data = self.tree.get_leaf(p_val)

            sample_idx[i] = idx
            sample_mem[i, :] = data

            prob = p / self.tree.total_p
            IS_weight[i, 0] = np.power(prob / min_prob, -self.beta)

        return sample_idx, sample_mem, IS_weight

    def batch_update(self, tree_idx, error):
        error += self.epsilon
        error = np.minimum(error, self.abs_err_upper)
        ps = np.power(error, self.alpha)
        for idx, p in zip(tree_idx, ps):
            self.tree.update(idx, p)