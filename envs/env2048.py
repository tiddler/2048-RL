from __future__ import print_function

from gym import spaces

import numpy as np
import gym
import random

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Env2048(gym.Env):

  def __init__(self, size=4):
    self._size = size
    self._tile_num = size * size
    self._action_set = {UP, DOWN, LEFT, RIGHT}
    # gym Env members
    self.action_space = spaces.Discrete(4)
    # observation
    self.observation_space = spaces.Box(
        low=0, high=2**self._tile_num, shape=(self._tile_num,), dtype=np.int)

    # initial the game
    self.reset()

  def step(self, action):
    if action not in self._action_set:
      raise Exception('Invalid Action Input')
    # in case users still send action after done without reset
    if self._done:
      self.reset()
    if self._can_move(action):
      reward = self._combine(action)
      self._add_tile()
      self._highest = np.max(self._board)
      self._score += reward
    else:
      reward = 0
    self._done = self._check_terminate()
    self._info = (self._score, self._available, self._highest)
    return self._board.flatten(), reward, self._done, self._info

  def reset(self):
    self._board = np.zeros((self._size, self._size), dtype=np.int)
    self._available = list(range(self._tile_num))
    self._score = 0
    self._done = False
    # add two numbers on the initial board
    self._add_tile()
    self._add_tile()
    self._highest = np.max(self._board)

  def render(self):
    print('Score: {}'.format(self._score))
    print('Highest: {}'.format(self._highest))
    print(self._board)

  def _add_tile(self):
    if not self._available:
      return
    idx = random.sample(self._available, 1)[0]
    num = 2 if random.random() < 0.9 else 4
    self._board[idx // self._size][idx % self._size] = num
    self._available.remove(idx)

  def _check_terminate(self):
    if self._available:
      return False
    for direction in [UP, DOWN, LEFT, RIGHT]:
      if self._can_move(direction):
        return False
    return True

  def _can_move(self, direction):
    """Check if the move direction is valid

    Args:
      direction(int): should be an int within [0, 3]

    Return:
      bool: is the movement is valid
    """
    if direction == LEFT:
      for i in range(self._size):
        for j in range(self._size - 1):
          if (self._board[i][j] == 0 and self._board[i][j + 1] > 0) or \
                  (self._board[i][j] != 0 and self._board[i][j] == self._board[i][j + 1]):
            return True
    elif direction == RIGHT:
      for i in range(self._size):
        for j in range(self._size - 1):
          if (self._board[i][j + 1] == 0 and self._board[i][j] > 0) or \
                  (self._board[i][j] != 0 and self._board[i][j] == self._board[i][j + 1]):
            return True
    elif direction == UP:
      for i in range(self._size):
        for j in range(self._size - 1):
          if (self._board[j][i] == 0 and self._board[j + 1][i]) or \
                  (self._board[j][i] != 0 and self._board[j][i] == self._board[j + 1][i]):
            return True
    elif direction == DOWN:
      for i in range(self._size):
        for j in range(self._size - 1):
          if (self._board[j + 1][i] == 0 and self._board[j][i]) or \
                  (self._board[j][i] != 0 and self._board[j + 1][i] == self._board[j][i]):
            return True
    return False

  def _combine(self, direction):
    reward = 0
    if direction == UP or direction == DOWN:
      for col in range(self._size):
        tiles = [x for x in self._board[:, col] if x]
        self._board[:, col] = 0
        i = 0
        if direction == DOWN:
          tiles = tiles[::-1]
        fill_index = 0 if direction == UP else self._size - 1
        fill_direction = 1 if direction == UP else -1
        while i < len(tiles):
          if i < len(tiles) - 1 and tiles[i] == tiles[i + 1]:
            new_num = tiles[i] + tiles[i + 1]
            reward += new_num
            self._board[fill_index, col] = new_num
            fill_index += fill_direction
            i += 2
          else:
            self._board[fill_index, col] = tiles[i]
            fill_index += fill_direction
            i += 1
    else:
      for row in range(self._size):
        tiles = [x for x in self._board[row, :] if x]
        self._board[row, :] = 0
        i = 0
        if direction == RIGHT:
          tiles = tiles[::-1]
        fill_index = 0 if direction == LEFT else self._size - 1
        fill_direction = 1 if direction == LEFT else -1
        while i < len(tiles):
          if i < len(tiles) - 1 and tiles[i] == tiles[i + 1]:
            new_num = tiles[i] + tiles[i + 1]
            reward += new_num
            self._board[row, fill_index] = new_num
            fill_index += fill_direction
            i += 2
          else:
            self._board[row, fill_index] = tiles[i]
            fill_index += fill_direction
            i += 1
    self._update_available()
    return reward

  def _update_available(self):
    self._available = [
        x for x in range(self._tile_num)
        if not self._board[x // self._size][x % self._size]
    ]
