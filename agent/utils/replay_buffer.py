import random
import numpy as np


class ReplayBuffer(object):

  def __init__(self, capacity):
    self.buffer = []
    self.capacity = capacity
    self.index = 0

  def __len__(self):
    return len(self.buffer)

  def push(self, *args):
    if len(self.buffer) < self.capacity:
      self.buffer.append(args)
    else:
      self.buffer[self.index] = args
    self.index = (self.index + 1) % self.capacity

  def sample(self, batch_size):
    samples = random.sample(self.buffer, batch_size)
    return map(lambda x: np.array(x, dtype=np.float32), zip(*samples))
