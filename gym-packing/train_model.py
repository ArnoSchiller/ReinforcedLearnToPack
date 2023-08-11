import os
import shutil
from typing import Callable

import numpy as np
import gymnasium

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, DQN,  A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import NormalActionNoise

import gym_packing
from gym_packing.callbacks import AdditionalTensorboardLogsCallback
import constants as c

TB_LOGS = os.path.join("training", "tb_logs")

i = 1
env_version = c.ENVIRONMENT.split("-")[-1]
model_name = f"p2d_{env_version}_{c.MODEL_NAME}"
MODEL_DIR = os.path.join(TB_LOGS, f"{model_name}_{i}")
while os.path.exists(MODEL_DIR):
    i += 1
    MODEL_DIR = os.path.join(TB_LOGS, f"{model_name}_{i}")
MODEL_DIR = os.path.join("training", f"{model_name}_{i}")

env = gymnasium.make(
    c.ENVIRONMENT,
    articles=c.ARTICLES,
    max_articles_per_order=c.MAX_ARTICLES,
    reward_strategies=c.REWARD_STRATEGIES,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP
)
env = FlattenObservation(env)
env = Monitor(env, f"{MODEL_DIR}/model")
obs = env.reset()
print(obs)

eval_env = gymnasium.make(
    c.ENVIRONMENT,
    articles=c.ARTICLES,
    max_articles_per_order=None,
    reward_strategies=c.REWARD_STRATEGIES,
    size=c.CONTAINER_SIZE,
    use_height_map=c.USE_HEIGHT_MAP
)
eval_env = FlattenObservation(eval_env)

eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                             log_path=MODEL_DIR+"/eval", eval_freq=5000)
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=-200, verbose=1)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODEL_DIR)

callback = CallbackList([
    AdditionalTensorboardLogsCallback(),
    # checkpoint_callback,
    eval_callback
])


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


learning_rate = linear_schedule(0.001)
n_actions = env.action_space.n
action_noise = NormalActionNoise(mean=np.zeros(
    n_actions), sigma=0.1 * np.ones(n_actions))

kwargs = {}
if c.MODEL_NAME == "PPO":
    MODEL = PPO
elif c.MODEL_NAME == "DQN":
    MODEL = DQN
    # kwargs["action_noise"] = action_noise
elif c.MODEL_NAME == "A2C":
    MODEL = A2C

model = MODEL(
    c.POLICY, env,
    # learning_rate=learning_rate,
    tensorboard_log=TB_LOGS
)

constants_fname = "constants.py"
shutil.copy(constants_fname, os.path.join(MODEL_DIR, constants_fname))

model.learn(
    total_timesteps=c.N_STEPS,
    progress_bar=True,
    tb_log_name=model_name,
    callback=callback
)
model.save(os.path.join(MODEL_DIR, "last_model.zip"))
