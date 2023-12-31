import gymnasium
import gym_packing

import constants as c
from packutils.data.article import Article

from gymnasium.wrappers import FlattenObservation


env = gymnasium.make(
    c.ENVIRONMENT,
    articles=[
        Article("", 20, 1, 5, 1),
        Article("", 30, 1, 5, 1),
        Article("", 40, 1, 5, 1),
    ],
    max_articles_per_order=None,
    max_next_items=0,  # c.MAX_NEXT_ITEMS,
    reward_strategies=c.REWARD_STRATEGIES,
    max_snappoints=10,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP,
    render_mode="human"
)
env = FlattenObservation(env)

print(env.reset())
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    # print(info)
