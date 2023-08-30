""" 
pip install fastapi "ray[serve]"
"""
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import os
import json
import gymnasium
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO


from packutils.data.article import Article
from packutils.data.packed_order import PackedOrder
from packutils.data.order import Order
import constants as c

app = FastAPI()

env = None
model = None


@app.on_event("startup")  # Code to be run when the server starts.
async def startup_event():

    global env
    env = gymnasium.make(
        c.ENVIRONMENT,
        articles=[],
        max_articles_per_order=None,
        reward_strategies=c.REWARD_STRATEGIES,
        size=c.CONTAINER_SIZE,
        use_height_map=c.USE_HEIGHT_MAP,
        run_expert=False,
        render_mode=None)
    env = FlattenObservation(env)

    global model
    model_weights = "student_10000_epochs_5000"
    model_path = os.path.join(os.path.dirname(__file__), model_weights)
    model = PPO.load(model_path)


class ArticleModel(BaseModel):
    id: str
    width: int
    length: int
    height: int
    amount: int


class OrderModel(BaseModel):
    order_id: str
    articles: List[ArticleModel]
    # supplies: List[Supply] = []


@app.get("/")
async def status():
    return {"status": "Healthy"}


@app.post("/model")
async def get_packing(orderModel: OrderModel):

    order = Order(
        order_id=orderModel.order_id,
        articles=[
            Article(
                article_id=a.id,
                width=a.width,
                length=a.length,
                height=a.height,
                amount=a.amount
            ) for a in orderModel.articles
        ]
    )

    done = False
    obs, _ = env.reset(options={"order": order})
    while not done:
        action, _states = model.predict(obs)
        print("Action:", action)
        obs, reward, done, _, info = env.step(action)
        print(reward, info)

    variant = env.packed_variant
    packed_order = PackedOrder(order_id=order.order_id)
    if variant is not None:
        packed_order.add_packing_variant(variant)

    print(packed_order)

    return packed_order.to_dict(as_string=False)
