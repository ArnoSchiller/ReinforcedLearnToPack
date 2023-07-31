import gymnasium
import gym_packing

import constants as c

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
# Parallel environments

env = gymnasium.make(c.ENVIRONMENT,
                     size=c.CONTAINER_SIZE,
                     use_height_map=True)
env = FlattenObservation(env)

obs = env.reset()
print(obs)

model = PPO("MlpPolicy", env, tensorboard_log="log")
model.learn(total_timesteps=2000000, progress_bar=True, tb_log_name='run2')
model.save("packing_2d_v0_test3")
