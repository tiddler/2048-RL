from __future__ import print_function

import numpy as np
import collections

class BaseAgent(object):

  def __init__(self, env):
    self._env = env
    self._action_space = env.action_space.n
    self._observation_space = env.observation_space.shape[0]

  def _get_action(self, state, is_test=False):
    return np.random.randint(0, self._action_space)

  def train(self, epochs=10, iterations=10, steps=1000):
    scores = []
    for e in range(epochs):
      for i in range(iterations):
        total_reward = 0
        state = self._env.reset()
        for t in range(steps):
          action = self._get_action(state)
          state, reward, done, info = self._env.step(action)
          total_reward += reward
          if done:
            scores.append((total_reward, info[-1]))
            state = self._env.reset()
            total_reward = 0
    return scores

  def test(self, render=False):
    score = 0
    state = self._env.reset()
    done = False
    while not done:
      action = self._get_action(state)
      state, reward, done, info = self._env.step(action)
      if render:
        self._env.render()
      score += reward
    print('final score: ', score)
    print('highest num: ', info[-1])

  def get_summary(self, scores):
    count = collections.Counter(x[1] for x in scores)
    return count
