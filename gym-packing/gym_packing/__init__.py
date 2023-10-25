from gymnasium.envs.registration import register

register(
    "gym_packing/Packing2DWorld-v1",
    entry_point="gym_packing.envs:Packing2DWorldEnvV1",
    max_episode_steps=300,
)

register(
    "gym_packing/Packing2DWorld-v2",
    entry_point="gym_packing.envs:Packing2DWorldEnvV2",
    max_episode_steps=300,
)

register(
    "gym_packing/Packing2DWorld-v3",
    entry_point="gym_packing.envs:Packing2DWorldEnvV3",
    max_episode_steps=300,
)

register(
    "gym_packing/Packing2DWorld-v4",
    entry_point="gym_packing.envs:Packing2DWorldEnvV4",
    max_episode_steps=300,
)
