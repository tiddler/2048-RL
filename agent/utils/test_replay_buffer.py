from __future__ import absolute_import

from . import replay_buffer
import unittest
import numpy as np


class TestReplayBuffer(unittest.TestCase):

  def test_replay(self):
    replay = replay_buffer.ReplayBuffer(100)
    for i in range(100):
      replay.push(i, i + 1, i + 2)
    sample1, sample2, sample3 = replay.sample(32)
    self.assertTrue(sample1.shape == sample2.shape == sample3.shape)
    np.testing.assert_array_equal(sample1, sample2 - 1)
    np.testing.assert_array_equal(sample1, sample3 - 2)

if __name__ == '__main__':
  unittest.main()
