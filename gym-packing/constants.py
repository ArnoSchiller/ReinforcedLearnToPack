

from gym_packing.envs.reward_strategies import RewardStrategy
from gym_packing.data.article import Article


ENVIRONMENT = "gym_packing/Packing2DWorld-v4"
CONTAINER_SIZE = (80, 80)
USE_HEIGHT_MAP = True
MAX_NEXT_ITEMS = 0
EXPERT_INTERACTIONS = 0  # 10_000

VERBOSE = 0
MODEL_NAME = "PPO"
POLICY = "MultiInputPolicy"
N_STEPS = 1_000_000
PRETRAINED_MODEL_PATH = None  # "training/p2d_v2_PPO_9/best_model.zip"


REWARD_STRATEGIES = [
    RewardStrategy.REWARD_ALL_COMPACTNESS
]

NUM_ENVS = 64

MAX_ARTICLES = 10

ARTICLES = [
    Article(
        article_id="article 1",
        width=30,
        length=1,
        height=30,
        amount=1
    ),
    Article(
        article_id="article 2",
        width=25,
        length=1,
        height=20,
        amount=2
    ),
    Article(
        article_id="article 3",
        width=25,
        length=1,
        height=10,
        amount=2
    ),
    Article(
        article_id="article 4",
        width=20,
        length=1,
        height=20,
        amount=4
    ),
    Article(
        article_id="article 5",
        width=20,
        length=1,
        height=15,
        amount=4
    ),
    Article(
        article_id="article 6",
        width=20,
        length=1,
        height=10,
        amount=4
    ),
    Article(
        article_id="article 7",
        width=20,
        length=1,
        height=5,
        amount=4
    ),
    Article(
        article_id="article 8",
        width=15,
        length=1,
        height=10,
        amount=5
    ),
    Article(
        article_id="article 9",
        width=15,
        length=1,
        height=5,
        amount=5
    ),
    Article(
        article_id="article 10",
        width=10,
        length=1,
        height=10,
        amount=5
    ),
    Article(
        article_id="article 11",
        width=10,
        length=1,
        height=5,
        amount=5
    ),
    Article(
        article_id="article 12",
        width=5,
        length=1,
        height=5,
        amount=5
    )
]


"""
ARTICLES = [
    Article(
        article_id="article 1",
        width=30,
        length=1,
        height=30,
        amount=1
    ),
    Article(
        article_id="article 2",
        width=20,
        length=1,
        height=10,
        amount=2
    ),
    Article(
        article_id="article 3",
        width=10,
        length=1,
        height=10,
        amount=4
    ),
    Article(
        article_id="article 4",
        width=10,
        length=1,
        height=5,
        amount=5
    )
]
"""
