from typing import Any, Iterator, Optional, Sequence, Tuple
from collections import deque, namedtuple
from pprint import pprint

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from animus import ICallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback

LOGDIR = "./logs_ddpg"


def set_requires_grad(model, requires_grad: bool):
    requires_grad = bool(requires_grad)
    for param in model.parameters():
        param.requires_grad = requires_grad


# Off-policy common

Transition = namedtuple(
    "Transition", field_names=["state", "action", "reward", "done", "next_state"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, size: int) -> Sequence[np.array]:
        indices = np.random.choice(
            len(self.buffer), size, replace=size > len(self.buffer)
        )
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool)
        next_states = np.array(next_states, dtype=np.float32)
        return states, actions, rewards, dones, next_states

    def __len__(self) -> int:
        return len(self.buffer)


# as far as RL does not have some predefined dataset,
# we need to specify epoch length by ourselfs
class ReplayDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, epoch_size: int = int(1e3)):
        self.buffer = buffer
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterator[Sequence[np.array]]:
        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.epoch_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], next_states[i]

    def __len__(self) -> int:
        return self.epoch_size


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Updates the `target` with the `source` smoothing by ``tau`` (inplace)."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# DDPG


class NormalizedActions(gym.ActionWrapper):
    def action(self, action: float) -> float:
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action: float) -> float:
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


def get_action(
    env, network: nn.Module, state: np.array, sigma: Optional[float] = None
) -> np.array:
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = network(state).detach().cpu().numpy()[0]
    if sigma is not None:
        action = np.random.normal(action, sigma)
    return action


def generate_session(
    env,
    network: nn.Module,
    sigma: Optional[float] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
) -> Tuple[float, int]:
    total_reward = 0
    state = env.reset()

    for t in range(env.spec.max_episode_steps):
        action = get_action(env, network, state=state, sigma=sigma)
        next_state, reward, done, _ = env.step(action)

        if replay_buffer is not None:
            transition = Transition(state, action, reward, done, next_state)
            replay_buffer.append(transition)

        total_reward += reward
        state = next_state
        if done:
            break

    return total_reward, t


def generate_sessions(
    env,
    network: nn.Module,
    sigma: Optional[float] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for i_episone in range(num_sessions):
        r, t = generate_session(
            env=env, network=network, sigma=sigma, replay_buffer=replay_buffer
        )
        sessions_reward += r
        sessions_steps += t
    return sessions_reward, sessions_steps


def get_network_actor(env):
    network = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
    )
    head = torch.nn.Sequential(nn.Linear(300, 1), nn.Tanh())
    return torch.nn.Sequential(network, head)


def get_network_critic(env):
    network = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0] + 1, 400),
        nn.LeakyReLU(0.01),
        nn.Linear(400, 300),
        nn.LeakyReLU(0.01),
    )
    head = nn.Linear(300, 1)
    return torch.nn.Sequential(network, head)


# Run


class ContinuousSamplerCallback(ICallback):
    def __init__(
        self,
        *,
        actor_attr: str,
        env,
        replay_buffer: ReplayBuffer,
        session_period: int,
        sigma: float,
        num_start_sessions: int = int(1e3),
        num_valid_sessions: int = int(1e2),
        prefix: str = "sampler",
    ):
        super().__init__()
        self.prefix = prefix
        self.env = env
        self.replay_buffer = replay_buffer
        self.session_period = session_period
        self.sigma = sigma
        self.actor_attr = actor_attr
        self.actor: nn.Module = None
        self.num_start_sessions = num_start_sessions
        self.num_valid_sessions = num_valid_sessions
        self.session_counter = 0
        self.session_steps = 0

    def on_experiment_start(self, exp: IExperiment) -> None:
        self.actor = getattr(exp, self.actor_attr)

        self.actor.eval()
        generate_sessions(
            env=self.env,
            network=self.actor,
            sigma=self.sigma,
            replay_buffer=self.replay_buffer,
            num_sessions=self.num_start_sessions,
        )
        self.actor.train()

    def on_epoch_start(self, exp: IExperiment):
        self.session_counter = 0
        self.session_steps = 0
        exp.epoch_metrics["sampler"] = {}

    def on_batch_end(self, exp: IExperiment):
        if exp.batch_step % self.session_period == 0:
            self.actor.eval()

            session_reward, session_steps = generate_session(
                env=self.env,
                network=self.actor,
                sigma=self.sigma,
                replay_buffer=self.replay_buffer,
            )

            self.session_counter += 1
            self.session_steps += session_steps

            exp.batch_metrics.update({f"{self.prefix}_train_reward": session_reward})
            exp.batch_metrics.update({f"{self.prefix}_train_steps": session_steps})

            self.actor.train()

    def on_epoch_end(self, exp: IExperiment):
        self.actor.eval()
        valid_rewards, valid_steps = generate_sessions(
            env=self.env, network=self.actor, num_sessions=int(self.num_valid_sessions)
        )
        self.actor.train()

        valid_rewards /= float(self.num_valid_sessions)
        valid_steps /= float(self.num_valid_sessions)
        exp.epoch_metrics[self.prefix]["sigma"] = self.sigma
        exp.epoch_metrics[self.prefix]["num_sessions"] = self.session_counter
        exp.epoch_metrics[self.prefix]["num_samples"] = self.session_steps
        exp.epoch_metrics[self.prefix]["updates_per_sample"] = (
            exp.dataset_sample_step / self.session_steps
        )
        exp.epoch_metrics[self.prefix]["valid_reward"] = valid_rewards
        exp.epoch_metrics[self.prefix]["valid_steps"] = valid_steps


class Experiment(IExperiment):
    def __init__(
        self,
        *,
        # trainer settings, ~training
        gamma: float,
        tau: float,
        tau_period: int = 1,
        # sampler settings, ~exploration
        session_period=100,  # in batches
        sigma: float = 0.3,
        # general
        num_epochs: int,
        env_name: str = "Pendulum-v0",
    ):
        super().__init__()
        self.num_epochs = num_epochs

        self.gamma: float = gamma
        self.tau: float = tau
        self.tau_period: int = tau_period

        self.session_period = session_period
        self.sigma = sigma

        self.env_name = env_name

        self.env = None
        self.replay_buffer = None
        self.actor: nn.Module = None
        self.critic: nn.Module = None
        self.target_actor: nn.Module = None
        self.target_critic: nn.Module = None
        self.actor_optimizer: nn.Module = None
        self.critic_optimizer: nn.Module = None

    def _setup_data(self):
        self.env = NormalizedActions(gym.make(self.env_name))

        self.batch_size = 64
        epoch_size = int(1e3) * self.batch_size
        buffer_size = int(1e5)
        self.replay_buffer = ReplayBuffer(buffer_size)
        train_loader = DataLoader(
            ReplayDataset(self.replay_buffer, epoch_size=epoch_size),
            batch_size=self.batch_size,
        )
        self.datasets = {"train": train_loader}

    def _setup_model(self):
        self.actor = get_network_actor(self.env)
        self.target_actor = get_network_actor(self.env)
        self.critic = get_network_critic(self.env)
        self.target_critic = get_network_critic(self.env)
        set_requires_grad(self.target_actor, requires_grad=False)
        set_requires_grad(self.target_critic, requires_grad=False)
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

        self.criterion = nn.MSELoss()

        lr_actor = 1e-4
        lr_critic = 1e-3
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def _setup_callbacks(self):
        # callback-based sampling is useful for exploration purposes
        # several samplers with different exploration setups could be used
        # to boost agent exploration
        self.callbacks = {
            "sampler": ContinuousSamplerCallback(
                actor_attr="actor",
                env=self.env,
                replay_buffer=self.replay_buffer,
                session_period=self.session_period,
                sigma=self.sigma,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="actor",
                logdir=LOGDIR,
                dataset_key="sampler",
                metric_key="valid_reward",
                minimize=False,
            ),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_data()
        self._setup_model()
        self._setup_callbacks()

    def on_dataset_start(self, exp: "IExperiment"):
        super().on_dataset_start(exp)
        self.dataset_metrics["critic_loss"] = list()
        self.dataset_metrics["actor_loss"] = list()

    def handle_batch(self, batch: Any) -> None:
        # model train/valid step
        states, actions, rewards, dones, next_states = batch

        # get actions for the current state
        pred_actions = self.actor(states)
        # get q-values for the actions in current states
        pred_critic_states = torch.cat([states, pred_actions], 1)
        # use q-values to train the actor model
        policy_loss = (-self.critic(pred_critic_states)).mean()

        with torch.no_grad():
            # get possible actions for the next states
            next_state_actions = self.target_actor(next_states)
            # get possible q-values for the next actions
            next_critic_states = torch.cat([next_states, next_state_actions], 1)
            next_state_values = self.target_critic(next_critic_states).detach().squeeze()
            next_state_values[dones] = 0.0

        # compute Bellman's equation value
        target_state_values = next_state_values * self.gamma + rewards
        # compute predicted values
        critic_states = torch.cat([states, actions], 1)
        state_values = self.critic(critic_states).squeeze()

        # train the critic model
        value_loss = self.criterion(state_values, target_state_values.detach())

        self.batch_metrics.update({"critic_loss": value_loss, "actor_loss": policy_loss})
        self.dataset_metrics["critic_loss"].append(value_loss.item())
        self.dataset_metrics["actor_loss"].append(policy_loss.item())

        if self.is_train_dataset:
            self.actor.zero_grad()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic.zero_grad()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            if self.batch_step % self.tau_period == 0:
                soft_update(self.target_actor, self.actor, self.tau)
                soft_update(self.target_critic, self.critic, self.tau)

    def on_dataset_end(self, exp: "IExperiment"):
        self.dataset_metrics["critic_loss"] = np.mean(
            self.dataset_metrics["critic_loss"]
        )
        self.dataset_metrics["actor_loss"] = np.mean(self.dataset_metrics["actor_loss"])
        super().on_dataset_end(exp)

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    # exp settings, ~training
    gamma = 0.99
    tau = 0.01
    tau_period = 1
    # callback, ~exploration
    session_period = 1
    sigma = 0.3
    # extras
    env_name = "Pendulum-v0"

    # train
    exp = Experiment(
        num_epochs=10,
        gamma=gamma,
        tau=tau,
        tau_period=tau_period,
        session_period=session_period,
        sigma=sigma,
        env_name=env_name,
    ).run()

    # evaluate
    env = gym.wrappers.Monitor(
        NormalizedActions(gym.make(env_name)), directory="videos_ddpg", force=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        f"{LOGDIR}/actor.best.pth", map_location=lambda storage, loc: storage
    )
    actor = get_network_actor(env)
    actor.load_state_dict(state_dict)
    rewards, _ = generate_sessions(env=env, network=actor.eval(), num_sessions=100)
    env.close()
    print("mean reward:", np.mean(rewards))
