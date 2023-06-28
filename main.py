import gym_examples
import gym
from gym.wrappers import FlattenObservation
from gym_examples.wrappers import RelativePosition

env = gym.make("gym_examples/GridWorld-v0", render_mode="human")
print(env.reset())
# env = FlattenObservation(env)
env = RelativePosition(env)

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    print(observation, reward, info)
