import random
import numpy as np
from collections import deque, namedtuple
from config import opt
from PriorMemory import ReplayBuffer3

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
# PriorTransition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'td_error'))


class ReplayMemory(object):
    # 这是经典的单记忆库方法，并且能够添加一条在线策略
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
    # 改进的双记忆库方法，可以添加一条以上的在线策略
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.on_buffer_size = 3  # 设定在线策略经验大小
        self.off_memory = []
        self.on_memory = []
        self.off_position = 0
        self.on_position = 0
        self.update = 0
        self.sample = []
        self.newest_push = 0

    def push(self, *args):
        # 在第一次初始化时，将离线策略记忆库填充至 16-on_buffer_size
        if len(self.off_memory) <= 16 - self.on_buffer_size:
            self.off_memory.append(None)
            self.off_memory[self.off_position] = Transition(*args)
            self.off_position += 1
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
            if len(self.off_memory) < self.buffer_size:
                self.off_memory.append(None)
            self.off_memory[self.off_position] = self.on_memory[i]
            # 先入先出规则，替换老元素
            self.off_position = (self.off_position + 1) % self.buffer_size

    def on_sample(self, size):
        # 随机采样离线策略经验
        self.sample = random.sample(self.off_memory, size-self.on_buffer_size)

        # 添加在线策略经验
        for i in range(self.on_buffer_size):
            self.sample.append(None)
            self.sample[len(self.sample)-1] = self.on_memory[i]

        batch = Transition(*zip(*self.sample))

        # 更新离线策略经验记忆库
        self.push_memory()

        return batch

    def __len__(self):
        return len(self.off_memory)+len(self.on_memory)


class PrioritizedReplayBuffer:
    # 优先经验回放方法
    def __init__(self, buffer_size, on_batch_size=6, batch_size=16, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.on_batch_size = on_batch_size  # 设定在线策略经验大小

        self.alpha = alpha  # controls the amount of prioritization
        self.beta = beta  # controls the amount of importance-sampling correction
        self.epsilon = epsilon  # a small constant to ensure non-zero priorities

        self.on_priorities = []
        self.off_priorities = []
        self.off_weights = []

        self.off_memory = []
        self.on_memory = []
        self.off_position = 0
        self.on_position = 0
        self.update = 0
        self.sample = []
        self.newest_push = 0

    def push(self, *args):
        self.newest_push = Transition(*args)

        # 在第一次初始化时，将离线策略记忆库填充至 batch_size-on_buffer_size
        if len(self.off_memory) < self.batch_size - self.on_batch_size:
            self.off_memory.append(None)
            self.off_priorities.append(None)
            self.off_weights.append(None)

            priority = (abs(self.newest_push.reward) + self.epsilon) ** self.alpha

            self.off_memory[self.off_position] = self.newest_push
            self.off_priorities[self.off_position] = priority
            self.off_weights[self.off_position] = (self.buffer_size * priority) ** (-self.beta)

            self.off_position += 1
        else:
            if len(self.on_memory) < self.on_batch_size:
                self.on_memory.append(None)
                self.on_priorities.append(None)

            priority = (abs(self.newest_push.reward) + self.epsilon) ** self.alpha

            self.on_priorities[self.on_position] = priority
            self.on_memory[self.on_position] = self.newest_push
            self.on_position += 1
            if self.on_position >= self.on_batch_size:
                # 开始软更新
                self.on_position = 0
                self.update = 1
            else:
                self.update = 0

    def push_memory(self):
        # 将上一次策略的经验存入离线策略
        for i in range(self.on_batch_size):
            if len(self.off_memory) < self.buffer_size:
                self.off_memory.append(None)
                self.off_priorities.append(None)
                self.off_weights.append(None)

            priority = (abs(self.on_memory[i].reward) + self.epsilon) ** self.alpha

            self.off_memory[self.off_position] = self.on_memory[i]
            self.off_priorities[self.off_position] = priority
            self.off_weights[self.off_position] = (self.buffer_size * priority) ** (-self.beta)

            self.off_position = (self.off_position + 1) % self.buffer_size  # 替换老元素

    def on_sample(self, batch_size):
        if self.__len__() < batch_size:
            return None

        priorities = np.array(self.off_priorities, dtype=np.float64)
        probs = priorities / priorities.sum()  # 归一化
        weights = np.array(self.off_weights, dtype=np.float64)
        weights /= weights.sum()

        # 优先经验方法采样离线策略经验
        indices = np.random.choice(len(self.off_memory), self.batch_size - self.on_batch_size, p=probs)  # 选取标号
        self.sample = [self.off_memory[i] for i in indices]

        # 添加在线策略经验
        for i in range(self.on_batch_size):
            self.sample.append(None)
            self.sample[len(self.sample)-1] = self.on_memory[i]

        is_weights = ((self.buffer_size * probs[indices]) ** (-self.beta)) / weights[indices]
        is_weights /= is_weights.max()

        batch = Transition(*zip(*self.sample))

        # 更新离线策略经验记忆库
        self.push_memory()

        return indices, batch, is_weights

    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.cpu().detach().numpy()
        # 获取对应的标签和error数组，更新对应的权重
        for i, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.off_priorities[i] = priority
            self.off_weights[i] = (self.buffer_size * priority) ** (-self.beta)

    def __len__(self):
        return len(self.off_memory)+len(self.on_memory)


def bubble_sort(x):
    length = len(x)
    sort_list = x
    for i in range(length):
        for j in range(1, length - i):
            if sort_list[j - 1] > sort_list[j]:
                sort_list[j], sort_list[j - 1] = sort_list[j - 1], sort_list[j]
    return sort_list


# replay_buffer = ReplayMemory(opt.buffer_size)
# replay_buffer = ReplayMemory1(opt.buffer_size)
# 注意修改在线策略的batch大小
replay_buffer = PrioritizedReplayBuffer(opt.buffer_size)
# replay_buffer = ReplayBuffer3(opt.buffer_size)
