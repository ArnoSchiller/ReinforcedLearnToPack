

from gym_packing.envs.reward_strategies import RewardStrategy


ENVIRONMENT = "gym_packing/Packing2DWorld-v2"
CONTAINER_SIZE = (40, 10)
USE_HEIGHT_MAP = True

VERBOSE = 0
MODEL_NAME = "PPO"
POLICY = "MlpPolicy"
N_STEPS = 20


REWARD_STRATEGIES = [
    RewardStrategy.REWARD_EACH_ITEM_PACKED_COMPACTNESS,
    RewardStrategy.REWARD_ALL_ITEMS_PACKED]
