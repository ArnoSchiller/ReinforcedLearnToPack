import gymnasium
import gym_packing

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO


model = PPO.load("packing_2d_v0_test")
env = gymnasium.make("gym_packing/Packing2DWorld-v0", render_mode="human")
env = FlattenObservation(env)

done = False
obs, info = env.reset()
print(obs)
while not done:
    action, _states = model.predict(obs)
    print("Action:", action)
    obs, reward, done, _, info = env.step(action)
    print(reward, info)
