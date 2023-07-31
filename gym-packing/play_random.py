import gymnasium
import gym_packing

from gymnasium.wrappers import FlattenObservation

env = gymnasium.make("gym_packing/Packing2DWorld-v0",
                     size=(10, 10),
                     use_height_map=True,
                     render_mode="human")
env = FlattenObservation(env)

print(env.reset())
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    print(observation, reward, info)
