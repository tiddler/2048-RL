from .utils.replay_buffer import ReplayBuffer

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

  def __init__(self, num_inputs, num_actions, ndim=128):
    super(DQN, self).__init__()
    self.layers = nn.Sequential(
        nn.Linear(num_inputs, ndim),
        nn.ReLU(),
        nn.Linear(ndim, ndim),
        nn.ReLU(),
        nn.Linear(ndim, num_actions))

  def forward(self, x):
    return self.layers(x)


class DQNAgent(object):

  def __init__(self, env, lr=0.005, buffer_size=50000, gamma=0.99):
    self._env = env
    self._action_space = env.action_space.n
    self._observation_space = env.observation_space.shape[0]
    self._use_cuda = torch.cuda.is_available()
    self._device = torch.device('cuda' if self._use_cuda else 'cpu')
    self._q_net = DQN(self._observation_space, self._action_space,
                      ndim=256).to(self._device)
    self._t_net = DQN(self._observation_space, self._action_space,
                      ndim=256).to(self._device)
    self._t_net.load_state_dict(self._q_net.state_dict())

    self._optimizer = optim.Adam(self._q_net.parameters(), lr=lr)
    self._loss_fn = nn.MSELoss()
    self._replay_buffer = ReplayBuffer(buffer_size)
    self._gamma = gamma

  def _get_action(self, state, epsilon=0.05, is_test=False):
    if not is_test and np.random.uniform() < epsilon:
      action = np.random.randint(0, self._action_space)
    else:
      with torch.no_grad():
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        action = self._q_net(state).argmax().numpy()
        action = int(action)
    return action

  def _random_action(self):
    return np.random.randint(0, self._action_space)

  def _update(self, batch_size):
    state, action, reward, next_state, done = self._replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(self._device)
    next_state = torch.FloatTensor(next_state).to(self._device)
    action = torch.LongTensor(action).to(self._device)
    reward = torch.FloatTensor(reward).to(self._device)
    done = torch.FloatTensor(done).to(self._device)

    q_values = self._q_net(state)
    next_q_values = self._t_net(
        next_state).detach()  # detach from graph, don't backpropagate
    optimal_q_values, _ = next_q_values.max(1)
    expected_q_values = reward + self._gamma * optimal_q_values * (1 - done)

    q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    loss = self._loss_fn(q_values, expected_q_values)
    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()
    return loss

  def train(self, num_step=10000, batch_size=64, update_period=100):
    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_decay = 5000
    epsilon_by_frame = lambda frame_idx: epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1 * frame_idx / epsilon_decay)

    scores = []
    loss_record = []
    state = self._env.reset()
    episode_reward = 0
    for i in range(1, num_step + 1):
      action = self._get_action(state, epsilon=epsilon_by_frame(i))
      next_state, reward, done, info = self._env.step(action)
      self._replay_buffer.push(state, action, reward, next_state, done)
      state = next_state
      episode_reward += reward

      if done:
        print('total_reward: ', episode_reward)
        scores.append((episode_reward, info[-1] if info else None))
        state = self._env.reset()
        episode_reward = 0

      if len(self._replay_buffer) > batch_size:
        loss = self._update(batch_size)
        loss_record.append(loss.data.item())

      if i % update_period == 0:
        self._t_net.load_state_dict(self._q_net.state_dict())

    return scores, loss_record

  def test(self, epoch=5, render=False):
    for _ in range(epoch):
      score = 0
      state = self._env.reset()
      done = False
      step = 0
      while not done:
        step += 1
        action = self._get_action(state, is_test=True)
        state, reward, done, info = self._env.step(action)
        if render:
          self._env.render()
        score += reward
      print('final score: ', score)
      if info:
        print('highest num: ', info[-1])

  def get_summary(self, scores):
    count = collections.Counter(x[1] for x in scores)
    return count
