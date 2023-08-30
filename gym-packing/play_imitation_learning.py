import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO, DQN

import gymnasium
from gymnasium.wrappers import FlattenObservation

from packutils.visual.packing_visualization import PackingVisualization
from packutils.data.article import Article
import constants as c


articles = [
    Article(
        article_id="article 3",
        width=10, length=1, height=10,
        amount=10
    ),
]

env = gymnasium.make(
    c.ENVIRONMENT,
    articles=articles,
    max_articles_per_order=None,
    max_next_items=c.MAX_NEXT_ITEMS,
    reward_strategies=c.REWARD_STRATEGIES,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP,
    run_expert=False,
    render_mode=None  # "human"
)
env = FlattenObservation(env)


model_weights = "student_5_rand_10000_epochs_5000"
model_path = os.path.join(os.path.dirname(__file__), model_weights)
model = PPO.load(model_path)

best_variant = None
best_compactness = None
for _ in range(10):
    done = False
    obs, _ = env.reset()
    total_reward = 0
    while not done:
        action, _states = model.predict(obs)
        print("Action:", action)
        obs, reward, done, _, info = env.step(action)
        print(reward, info)
        total_reward += reward

    compactness = env.calculate_compactness()
    if best_compactness is None or compactness > best_compactness:
        best_compactness = compactness
        best_variant = env.packed_variant

print(best_compactness, best_variant)
vis = PackingVisualization()
vis.visualize_bin(best_variant.bins[0])

# img = env.render("rgb_array")
# plt.show(img)
