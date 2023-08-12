

from gym_packing.envs.reward_strategies import RewardStrategy
from packutils.data.article import Article


ENVIRONMENT = "gym_packing/Packing2DWorld-v2"
CONTAINER_SIZE = (40, 10)
USE_HEIGHT_MAP = True

VERBOSE = 0
MODEL_NAME = "PPO"
POLICY = "MultiInputPolicy"
N_STEPS = 1_000_000


REWARD_STRATEGIES = [
    RewardStrategy.REWARD_EACH_ITEM_PACKED,
    RewardStrategy.PENALIZE_EACH_ITEM_DISTANCE_COG,
    RewardStrategy.PENALIZE_PACKING_FAILED,
    RewardStrategy.REWARD_COMPACTNESS]

NUM_ENVS = 32

MAX_ARTICLES = 20
ARTICLES = [
    Article(
        article_id="article large",
        width=4,
        length=1,
        height=4,
        amount=15
    ),
    Article(
        article_id="article small",
        width=2,
        length=1,
        height=2,
        amount=15
    )
]
