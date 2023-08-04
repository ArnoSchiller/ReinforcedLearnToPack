import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.logger import Image


class AdditionalTensorboardLogsCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals["dones"]):

            info = self.locals["infos"][idx]
            packed_items_percentage = info["packed"] / \
                (info["packed"] + info["remaining"])

            self.logger.record(f"packed_items_percentage_{idx}",
                               packed_items_percentage)
            if done:
                self.logger.record(f"done_packed_items_percentage_{idx}",
                                   packed_items_percentage)

            self.logger.dump(step=self.n_calls)
        return True
