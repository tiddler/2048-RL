from .env2048 import Env2048
from gym.envs.registration import register

register(
  id='2048-v0',
  entry_point='envs.env2048:Env2048'
)
