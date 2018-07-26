import gym
import envs

if __name__ == '__main__':
  env = gym.make('2048-v0')
  done = False
  while not done:
    env.render()
    action = input('enter action: u: up, j: down, h: left, k: right:  ')
    if action == 'u':
      action = 0
    elif action == 'j':
      action = 1
    elif action == 'h':
      action = 2
    else:
      action = 3
    state, reward, done, info = env.step(action)
  input('finish! Press Enter to exit')