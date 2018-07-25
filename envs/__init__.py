from .game2048_env import Game2048Env
from gym.envs.registration import register

register(
    id='2048-v0',
    entry_point='envs.game2048_env:Game2048Env'
)
