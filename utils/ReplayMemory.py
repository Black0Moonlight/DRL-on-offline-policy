import random
import numpy as np
from collections import deque, namedtuple
from config import opt

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    # 这是经典的单记忆库方法，能够添加一条在线策略
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0
        self.newest_push = 0

    def push(self, *args):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.newest_push = Transition(*args)
        self.memory[self.position] = self.newest_push
        self.position = (self.position + 1) % self.buffer_size  # 替换老元素

    def sample(self, size):
        # off-line
        samples = random.sample(self.memory, size)
        batch = Transition(*zip(*samples))
        return batch

    def on_sample(self, size):
        # off-line
        samples = random.sample(self.memory, size)

        # on-line
        samples[0] = self.newest_push  # 将最新的buffer放入
        batch = Transition(*zip(*samples))
        return batch

    def __len__(self):
        return len(self.memory)


class ReplayMemory1(object):
    # 这是改进的双记忆库方法
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.on_buffer_size = 2  # 每X步进行一次软更新
        self.memory = []
        self.on_memory = []
        self.position = 0
        self.on_position = 0
        self.update = 0
        self.sample = []

    def push(self, *args):
        if len(self.memory) <= 16:
            self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position += 1
        else:
            if len(self.on_memory) < self.on_buffer_size:
                self.on_memory.append(None)
            self.on_memory[self.on_position] = Transition(*args)
            self.on_position += 1
            if self.on_position >= self.on_buffer_size:
                # 开始软更新
                self.on_position = 0
                self.update = 1
                return 1
            else:
                self.update = 0
                return 0

    def push_memory(self):
        # 将上一次策略的经验存入离线策略
        for i in range(self.on_buffer_size):
            if len(self.memory) < self.buffer_size:
                self.memory.append(None)
            self.memory[self.position] = self.on_memory[i]
            self.position = (self.position + 1) % self.buffer_size  # 替换老元素

    def on_sample(self, size):
        # off-line
        self.sample = random.sample(self.memory, size-self.on_buffer_size)

        # on-line
        for i in range(self.on_buffer_size):
            self.sample.append(None)
            self.sample[len(self.sample)-1] = self.on_memory[i]

        batch = Transition(*zip(*self.sample))
        self.push_memory()
        return batch

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    # 优先经验回放方法
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.buffer_size = buffer_size
        self.alpha = alpha  # controls the amount of prioritization
        self.beta = beta  # controls the amount of importance-sampling correction
        self.epsilon = epsilon  # a small constant to ensure non-zero priorities

        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.weights = deque(maxlen=buffer_size)
        self.newest_push = 0

    def push(self, state, action, reward, next_state, done):
        experience = Transition(state, action, reward, next_state, done)
        self.newest_push = experience

        # Calculate the priority for this experience
        priority = (abs(reward) + self.epsilon) ** self.alpha
        self.memory.append(experience)
        self.priorities.append(priority)
        self.weights.append((self.buffer_size * priority) ** (-self.beta))

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        weights = np.array(self.weights)
        probs /= weights.sum()
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        batch = [self.memory[i] for i in indices]
        # weights = torch.tensor(weights[indices], dtype=torch.float32)

        # is_weights = ((self.buffer_size * probs[indices]) ** (-self.beta)) / weights
        # is_weights /= is_weights.max()

        batch = Transition(*zip(*batch))

        return batch

    def on_sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        weights = np.array(self.weights)
        probs /= weights.sum()
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        batch = [self.memory[i] for i in indices]

        batch[0] = self.newest_push

        batch = Transition(*zip(*batch))
        return batch

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[i] = priority
            self.weights[i] = (self.buffer_size * priority) ** (-self.beta)

    def __len__(self):
        return len(self.memory)


def bubble_sort(x):
    length = len(x)
    sort_list = x
    for i in range(length):
        for j in range(1, length - i):
            if sort_list[j - 1] > sort_list[j]:
                sort_list[j], sort_list[j - 1] = sort_list[j - 1], sort_list[j]
    return sort_list


replay_buffer = ReplayMemory(opt.buffer_size)
# replay_buffer = ReplayMemory1(opt.buffer_size)
# replay_buffer = PrioritizedReplayBuffer(opt.buffer_size)
