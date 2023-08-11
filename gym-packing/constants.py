

from gym_packing.envs.reward_strategies import RewardStrategy
from packutils.data.article import Article


ENVIRONMENT = "gym_packing/Packing2DWorld-v2"
CONTAINER_SIZE = (40, 10)
USE_HEIGHT_MAP = True

VERBOSE = 0
MODEL_NAME = "PPO"
POLICY = "MlpPolicy"
N_STEPS = 20_000_000


REWARD_STRATEGIES = [
    RewardStrategy.REWARD_EACH_ITEM_PACKED_COMPACTNESS,
    RewardStrategy.REWARD_ALL_ITEMS_PACKED]


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
