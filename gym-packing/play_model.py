import os
import gymnasium
import gym_packing

import constants as c

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, DQN


model_dir = "p2d_v2_PPO_12"
model_weights = "best_model"
model_path = os.path.join("training", model_dir, model_weights)

if model_dir.count("PPO") > 0:
    model = PPO.load(model_path)
elif model_dir.count("DQN") > 0:
    model = DQN.load("packing_2d_v2_DQN")

env = gymnasium.make(
    c.ENVIRONMENT,
    articles=c.ARTICLES,
    max_articles_per_order=None,
    reward_strategies=c.REWARD_STRATEGIES,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP,
    render_mode="human")
# env = FlattenObservation(env)

done = False
obs, info = env.reset()
print(obs)

while not done:
    action, _states = model.predict(obs)
    print("Action:", action)
    obs, reward, done, _, info = env.step(action)
    print(reward, info)
