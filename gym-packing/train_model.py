import gymnasium
import gym_packing

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
# Parallel environments

env = gymnasium.make("gym_packing/Packing2DWorld-v0")  # , render_mode="human")
env = FlattenObservation(env)

obs = env.reset()
print(obs)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=20000)
model.save("packing_2d_v0_test")
