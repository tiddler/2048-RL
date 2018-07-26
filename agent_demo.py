from agent.base_agent import BaseAgent
import envs
import gym

if __name__ == '__main__':
  env = gym.make('2048-v0')
  agent = BaseAgent(env)
  train_log = agent.train()
  test_result = agent.test()