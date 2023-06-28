from gymnasium.envs.registration import register

register(
    "gym_packing/Packing2DWorld-v0",
    entry_point="gym_packing.envs:Packing2DWorldEnv",
    max_episode_steps=300,
)
