from __future__ import absolute_import

from . import env2048
import unittest
import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class TestBoard(unittest.TestCase):

  def test_combine(self):
    env = env2048.Env2048()
    env._board = np.array([
      [2, 2, 2, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0]
    ])
    reward = env._combine(LEFT)
    np.testing.assert_array_equal(env._board, np.array([
      [4, 2, 0, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0]
    ]))
    self.assertEqual(reward, 4)

    env._board = np.array([
      [2, 2, 2, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0]
    ])
    reward = env._combine(RIGHT)
    np.testing.assert_array_equal(env._board, np.array([
      [0, 0, 2, 4],
      [0, 0, 0, 2],
      [0, 0, 0, 2],
      [0, 0, 0, 0]
    ]))
    self.assertEqual(reward, 4)

    env._board = np.array([
      [2, 2, 2, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0]
    ])
    reward = env._combine(UP)
    np.testing.assert_array_equal(env._board, np.array([
      [4, 2, 2, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ]))
    self.assertEqual(reward, 4)

    env._board = np.array([
      [2, 2, 2, 0],
      [2, 0, 0, 0],
      [2, 0, 0, 0],
      [0, 0, 0, 0]
    ])
    reward = env._combine(DOWN)
    np.testing.assert_array_equal(env._board, np.array([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [2, 0, 0, 0],
      [4, 2, 2, 0]
    ]))
    self.assertEqual(reward, 4)

    env._board = np.array([
      [2, 2, 2, 16],
      [0, 0, 0, 8],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ])
    reward = env._combine(RIGHT)
    np.testing.assert_array_equal(env._board, np.array([
      [0, 2, 4, 16],
      [0, 0, 0, 8],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ]))
    self.assertEqual(reward, 4)

  def test_terminate(self):
    env = env2048.Env2048()
    env._board = np.array([
      [2, 2, 2, 16],
      [0, 0, 0, 8],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ])
    self.assertEqual(env._check_terminate(), False)

    env._board = np.array([
      [2, 4, 2, 4],
      [4, 2, 4, 2],
      [2, 4, 2, 4],
      [4, 2, 4, 2]
    ])
    env._update_available()
    self.assertEqual(env._check_terminate(), True)


if __name__ == '__main__':
  unittest.main()
