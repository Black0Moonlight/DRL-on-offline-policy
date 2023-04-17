import random
import numpy as np


class ReplayMemory(object):
    def __init__(self, capacity, labels):
        self.capacity = capacity
        self.transition = labels
        self.memory = []
        self.list = []
        self.position = 0
        self.newest_push = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.newest_push = self.transition(*args)
        self.memory[self.position] = self.newest_push
        self.position = (self.position + 1) % self.capacity  # 替换老元素

    def sample(self, size):
        # off-line
        return random.sample(self.memory, size)

    def on_sample(self, size):
        # off-line
        x = random.sample(self.memory, size)

        # on-line
        x[0] = self.newest_push  # 将最新的buffer放入
        return x

    def opt_sample(self, size):  # 未完成
        # off-line
        x = bubble_sort(self.memory)  # 对于好的记忆给以大的优先级
        x = x[:size]

        # on-line
        x[0] = self.newest_push
        return x

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
