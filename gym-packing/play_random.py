import gymnasium
import gym_packing

import constants as c

from gymnasium.wrappers import FlattenObservation


env = gymnasium.make(
    c.ENVIRONMENT,
    articles=c.ARTICLES,
    max_articles_per_order=None,
    reward_strategies=c.REWARD_STRATEGIES,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP,
    render_mode="human")
env = FlattenObservation(env)

print(env.reset())
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    print(observation, reward, info)
