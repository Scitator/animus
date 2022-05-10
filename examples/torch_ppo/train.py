import time

import gym
import tensorboardX
import torch

from animus import IExperiment, set_global_seed
from animus.torch.callbacks import TorchCheckpointerCallback

from src.acmodel import ACModel
from src.settings import LOGDIR
from src.trainer import A2CTrainer, PPOTrainer
from src.utils import synthesize

TRAINERS = {
    "a2c": A2CTrainer,
    "ppo": PPOTrainer,
}


class Experiment(IExperiment):
    def __init__(
        self,
        *,
        # general
        num_steps: int,
        num_envs: int,
        alg_name: str = "ppo",
        env_name: str = "CartPole-v1",
        recurrent: bool = False,
        **trainer_kwargs,
    ):
        super().__init__()
        self.update_step = 0
        self.num_epochs = num_steps
        self.num_envs = num_envs
        self.recurrent = recurrent
        self.env_name = env_name
        self.alg_name = alg_name
        self.trainer_kwargs = trainer_kwargs

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        envs = []
        for i in range(self.num_envs):
            envs.append(gym.make(self.env_name))
        observation_space, action_space = envs[0].observation_space, envs[0].action_space
        self.acmodel = ACModel(
            observation_space=observation_space,
            action_space=action_space,
            recurrent=self.recurrent,
        )
        self.trainer = TRAINERS[self.alg_name](
            envs=envs, acmodel=self.acmodel, device=self.device, **self.trainer_kwargs
        )
        self.callbacks = {
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="acmodel",
                logdir=LOGDIR,
                metric_key="return_mean",
                minimize=False,
            ),
        }
        self.txt_logger = get_txt_logger(LOGDIR)
        self.tb_logger = tensorboardX.SummaryWriter(LOGDIR)

    def on_epoch_start(self, exp: "IExperiment"):
        self.epoch_metrics = {}
        set_global_seed(self.seed + self.epoch_step)

    def run_epoch(self) -> None:
        update_start_time = time.time()
        exps, logs1 = self.trainer.collect_experiences()
        logs2 = self.trainer.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        self.update_step += 1
        self.epoch_step += logs["num_steps"]
        fps = logs["num_steps"] / (update_end_time - update_start_time)

        return_per_episode = synthesize(logs["return_per_episode"])
        rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"])
        num_steps_per_episode = synthesize(logs["num_steps_per_episode"])

        header = ["update", "steps", "FPS"]
        data = [self.update_step, self.epoch_step, fps]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_steps_" + key for key in num_steps_per_episode.keys()]
        data += num_steps_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [
            logs["entropy"],
            logs["value"],
            logs["policy_loss"],
            logs["value_loss"],
            logs["grad_norm"],
        ]
        self.txt_logger.info(
            "U {} | S {:06} | FPS {:04.0f} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}".format(
                *data
            )
        )
        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        self.epoch_metrics = dict(zip(header, data))
        for field, value in self.epoch_metrics.items():
            self.tb_logger.add_scalar(field, value, self.epoch_step)


if __name__ == "__main__":
    exp = Experiment(
        num_steps=int(1e6),
        num_envs=16,
        env_name="LunarLander-v2",  # "CartPole-v1",
        recurrent=False,
        alg_name="ppo",
        # ppo
        num_steps_per_proc=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=1,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
    ).run()
