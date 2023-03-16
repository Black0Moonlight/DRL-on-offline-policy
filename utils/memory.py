import numpy as np
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 256


class ReplayBuffer:

    def __init__(self, a_dim, s_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.capacity = REPLAY_BUFFER_SIZE
        self.buffer = np.zeros((self.capacity, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.position = 0    # index
        self.trans_num = 0      # 边填边训练
        self.buffer_full = False

    def push(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.position % self.capacity  # replace the old memory with new memory
        self.buffer[index, :] = transition
        self.position += 1
        if self.position >= self.capacity:      # indicator for learning
            self.buffer_full = True
        self.trans_num = min(self.position, self.capacity)      # 边填边训练

    def sample(self):
        # indices = np.random.choice(self.capacity, BATCH_SIZE)
        indices = np.random.choice(self.trans_num, BATCH_SIZE)
        bt = self.buffer[indices, :]
        state = bt[:, :self.s_dim]
        action = bt[:, self.s_dim: self.s_dim + self.a_dim]
        reward = bt[:, self.s_dim + self.a_dim: self.s_dim + self.a_dim + 1]
        next_state = bt[:, -self.s_dim - 1: -1]
        done = bt[:, -1]

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
