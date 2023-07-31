import gymnasium
import gym_packing

import constants as c

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, DQN
# Parallel environments

env = gymnasium.make(c.ENVIRONMENT,
                     size=c.CONTAINER_SIZE,
                     use_height_map=True)
env = FlattenObservation(env)
env_version = c.ENVIRONMENT.split("-")[-1]
obs = env.reset()
print(obs)

# model = PPO("MlpPolicy", env, tensorboard_log="log")
model = DQN("MlpPolicy", env, tensorboard_log="log")
model_name = f"{model.__class__.__name__}"
model.learn(total_timesteps=2000000, progress_bar=True, tb_log_name=model_name)
model.save(f"packing_2d_{env_version}_{model_name}")
