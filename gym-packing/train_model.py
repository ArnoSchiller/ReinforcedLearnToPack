import os
import gymnasium
import gym_packing

import constants as c

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, DQN

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from gym_packing.callbacks import AdditionalTensorboardLogsCallback
from stable_baselines3.common.monitor import Monitor

TB_LOGS = os.path.join("training", "tb_logs")

i = 1
env_version = c.ENVIRONMENT.split("-")[-1]
model_name = f"p2d_{env_version}_{c.MODEL_NAME}"
MODEL_DIR = os.path.join("training", f"{model_name}_{i}")
while os.path.exists(MODEL_DIR):
    i += 1
    MODEL_DIR = os.path.join("training", f"{model_name}_{i}")

env = gymnasium.make(
    c.ENVIRONMENT,
    size=c.CONTAINER_SIZE,
    use_height_map=True)
env = FlattenObservation(env)
env = Monitor(env, f"{MODEL_DIR}/model")
obs = env.reset()
print(obs)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODEL_DIR)
eval_env = gymnasium.make(
    c.ENVIRONMENT, size=c.CONTAINER_SIZE, use_height_map=True)
eval_env = FlattenObservation(eval_env)
eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR,
                             log_path=MODEL_DIR+"/eval", eval_freq=5000)
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=-200, verbose=1)

callback = CallbackList([
    AdditionalTensorboardLogsCallback(),
    checkpoint_callback,
    eval_callback
])

if c.MODEL_NAME == "PPO":
    model = PPO("MlpPolicy", env, tensorboard_log=TB_LOGS)
elif c.MODEL_NAME == "DQN":
    model = DQN("MlpPolicy", env, tensorboard_log=TB_LOGS)

model.learn(
    total_timesteps=c.N_STEPS,
    progress_bar=True,
    tb_log_name=model_name,
    callback=callback)


model.save(os.path.join(MODEL_DIR, "last_model.zip"))
