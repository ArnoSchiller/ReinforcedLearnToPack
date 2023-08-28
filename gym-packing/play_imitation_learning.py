from torch.utils.data.dataset import Dataset, random_split
import gymnasium
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, DQN

from gymnasium.wrappers import FlattenObservation

import constants as c
import os

from packutils.data.article import Article

articles = [
    Article(
        article_id="article 1",
        width=30, length=1, height=30,
        amount=1
    ),
    Article(
        article_id="article 2",
        width=20, length=1, height=10,
        amount=2
    ),
    Article(
        article_id="article 3",
        width=10, length=1, height=10,
        amount=4
    ),
    Article(
        article_id="article 4",
        width=10, length=1, height=5,
        amount=5
    )
]

env = gymnasium.make(
    c.ENVIRONMENT,
    articles=articles,
    max_articles_per_order=None,
    reward_strategies=c.REWARD_STRATEGIES,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP,
    run_expert=False,
    render_mode="human")
env = FlattenObservation(env)


model_weights = "student_10000_epochs_5000"
model_path = os.path.join(os.path.dirname(__file__), model_weights)
model = PPO.load(model_path)

done = False
obs, _ = env.reset()
while not done:
    action, _states = model.predict(obs)
    print("Action:", action)
    obs, reward, done, _, info = env.step(action)
    print(reward, info)
