import os
import random
import numpy as np


# 初始化：需要输入memory的容量：entry_size，初始化的代码如下：
class ReplayMemory:
    def __init__(self, entry_size):
        self.entry_size = entry_size
        self.memory_size = 204800
        self.actions = np.empty(self.memory_size, dtype=np.int64)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float32)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float32)
        self.batch_size = 2048
        self.count = 0
        self.current = 0

    # 添加（s, a）对：add(self, prestate, poststate, reward, action)，
    # 从add方法的参数可以看出参数包括：（上一个状态，下一个状态，奖励，动作），代码如下：
    def add(self, prestate, poststate, action, reward):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    # 采样：sample(self)，经过多次add后，每个agent已经有了多个（s, a）对，但是实际训练的时候一次取出batch_size个（s, a）对进行训练，代码如下所示：
    def sample(self):

        if self.count < self.batch_size:
            # range代表从0到self.count
            number = self.count
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.batch_size)
            number = self.batch_size
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]

        return prestate, poststate, actions, rewards, number
