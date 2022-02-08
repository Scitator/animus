from typing import Iterator, Optional, Sequence, Tuple
from collections import deque, namedtuple
from pprint import pprint

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from animus import ICallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback

LOGDIR = "./logs_reinforce"


def set_requires_grad(model, requires_grad: bool):
    requires_grad = bool(requires_grad)
    for param in model.parameters():
        param.requires_grad = requires_grad


# On-policy common

Rollout = namedtuple("Rollout", field_names=["states", "actions", "rewards"])


class RolloutBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, rollout: Rollout):
        self.buffer.append(rollout)

    def sample(self, idx: int) -> Sequence[np.array]:
        states, actions, rewards = self.buffer[idx]
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        return states, actions, rewards

    def __len__(self) -> int:
        return len(self.buffer)


# as far as RL does not have some predefined dataset,
# we need to specify epoch length by ourselfs
class RolloutDataset(IterableDataset):
    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer

    def __iter__(self) -> Iterator[Sequence[np.array]]:
        for i in range(len(self.buffer)):
            states, actions, rewards = self.buffer.sample(i)
            yield states, actions, rewards
        self.buffer.buffer.clear()

    def __len__(self) -> int:
        return self.buffer.capacity


# REINFORCE


def get_cumulative_rewards(rewards, gamma=0.99):
    G = [rewards[-1]]
    for r in reversed(rewards[:-1]):
        G.insert(0, r + gamma * G[0])
    return G


def to_one_hot(y, n_dims=None):
    """Takes an integer vector and converts it to 1-hot matrix."""
    y_tensor = y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot


def get_action(env, network: nn.Module, state: np.array) -> int:
    state = torch.tensor(state[None], dtype=torch.float32)
    logits = network(state).detach()
    probas = F.softmax(logits, -1).cpu().numpy()[0]
    action = np.random.choice(len(probas), p=probas)
    return int(action)


def generate_session(
    env,
    network: nn.Module,
    t_max: int = 1000,
    rollout_buffer: Optional[RolloutBuffer] = None,
) -> Tuple[float, int]:
    total_reward = 0
    states, actions, rewards = [], [], []
    state = env.reset()

    for _ in range(t_max):
        action = get_action(env, network, state=state)
        next_state, reward, done, _ = env.step(action)

        # record session history to train later
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        total_reward += reward
        state = next_state
        if done:
            break
    if rollout_buffer is not None:
        rollout_buffer.append(Rollout(states, actions, rewards))

    return total_reward, len(states)


def generate_sessions(
    env,
    network: nn.Module,
    t_max: int = 1000,
    rollout_buffer: Optional[RolloutBuffer] = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for _ in range(int(num_sessions)):
        r, t = generate_session(
            env=env, network=network, t_max=t_max, rollout_buffer=rollout_buffer
        )
        sessions_reward += r
        sessions_steps += t
    return sessions_reward, sessions_steps


def get_network(env, num_hidden: int = 128):
    network = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0], num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_hidden),
        nn.ReLU(),
    )
    head = nn.Linear(num_hidden, env.action_space.n)
    return torch.nn.Sequential(network, head)


# Run


class SoftmaxSamplerCallback(ICallback):
    def __init__(
        self,
        *,
        actor_attr: str,
        env,
        rollout_buffer: RolloutBuffer,
        num_train_sessions: int = 1e2,
        num_valid_sessions: int = 1e2,
        prefix: str = "sampler",
    ):
        super().__init__()
        self.prefix = prefix
        self.actor_attr = actor_attr
        self.actor: nn.Module = None
        self.env = env
        self.rollout_buffer = rollout_buffer
        self.num_train_sessions = num_train_sessions
        self.num_valid_sessions = num_valid_sessions

    def on_epoch_start(self, exp: IExperiment):
        self.actor = getattr(exp, self.actor_attr)

        self.actor.eval()
        train_rewards, train_steps = generate_sessions(
            env=self.env,
            network=self.actor,
            rollout_buffer=self.rollout_buffer,
            num_sessions=self.num_train_sessions,
        )
        train_rewards /= float(self.num_train_sessions)
        train_steps /= float(self.num_train_sessions)
        exp.epoch_metrics[self.prefix] = {}
        exp.epoch_metrics[self.prefix]["train_reward"] = train_rewards
        exp.epoch_metrics[self.prefix]["train_steps"] = train_steps
        self.actor.train()

    def on_epoch_end(self, exp: IExperiment):
        self.actor.eval()
        valid_rewards, valid_steps = generate_sessions(
            env=self.env, network=self.actor, num_sessions=self.num_valid_sessions
        )
        self.actor.train()

        valid_rewards /= float(self.num_valid_sessions)
        valid_steps /= float(self.num_valid_sessions)
        exp.epoch_metrics[self.prefix]["valid_reward"] = valid_rewards
        exp.epoch_metrics[self.prefix]["valid_steps"] = valid_steps


class Experiment(IExperiment):
    def __init__(
        self,
        *,
        gamma: float,
        entropy_coef: float = 0.1,
        # general
        num_epochs: int,
        env_name: str = "CartPole-v1",
    ):
        super().__init__()
        self.num_epochs = num_epochs

        self.gamma: float = gamma
        self.entropy_coef: float = entropy_coef

        self.env_name = env_name

        self.env = None
        self.rollout_buffer = None
        self.network: nn.Module = None
        self.target_network: nn.Module = None

    def _setup_data(self):
        self.env = gym.make(self.env_name)

        self.batch_size = 1  # in trajectories
        buffer_size = int(1e2)
        self.rollout_buffer = RolloutBuffer(buffer_size)
        train_loader = DataLoader(
            RolloutDataset(self.rollout_buffer), batch_size=self.batch_size
        )
        self.datasets = {"train": train_loader}

    def _setup_model(self):
        self.model = get_network(self.env).to(self.device)
        self.criterion = nn.MSELoss()
        lr = 3e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _setup_callbacks(self):
        # callback-based sampling is useful for exploration purposes
        # several samplers with different exploration setups could be used
        # to boost agent exploration
        self.callbacks = {
            "sampler": SoftmaxSamplerCallback(
                actor_attr="model",
                env=self.env,
                rollout_buffer=self.rollout_buffer,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="model",
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
        self.dataset_metrics["loss"] = list()

    def run_batch(self):
        # model train/valid step
        # ATTENTION:
        #   because of different trajectories lens
        #   ONLY batch_size==1 supported
        states, actions, rewards = self.batch
        states, actions, rewards = states[0], actions[0], rewards[0]
        cumulative_returns = torch.tensor(get_cumulative_rewards(rewards, gamma))

        logits = self.model(states)
        probas = F.softmax(logits, -1)
        logprobas = F.log_softmax(logits, -1)
        n_actions = probas.shape[1]
        logprobas_for_actions = torch.sum(
            logprobas * to_one_hot(actions, n_dims=n_actions), dim=1
        )

        J_hat = torch.mean(logprobas_for_actions * cumulative_returns)
        entropy_reg = -torch.mean(torch.sum(probas * logprobas, dim=1))
        loss = -J_hat - self.entropy_coef * entropy_reg

        self.batch_metrics.update({"loss": loss})
        self.dataset_metrics["loss"].append(loss.item())

        if self.is_train_dataset:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_dataset_end(self, exp: "IExperiment"):
        self.dataset_metrics["loss"] = np.mean(self.dataset_metrics["loss"])
        super().on_dataset_end(exp)

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    # exp settings, ~training
    gamma = 0.99
    # extras
    env_name = "CartPole-v1"

    exp = Experiment(
        num_epochs=30,
        gamma=gamma,
        env_name=env_name,
    ).run()

    try:
        env = gym.wrappers.Monitor(
            gym.make(env_name), directory="videos_reinforce", force=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(
            f"{LOGDIR}/model.best.pth", map_location=lambda storage, loc: storage
        )
        actor = get_network(env)
        actor.load_state_dict(state_dict)
        rewards, _ = generate_sessions(env=env, network=actor.eval(), num_sessions=100)
        env.close()
        print("mean reward:", np.mean(rewards))
    except Exception:
        env = gym.make(env_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(
            f"{LOGDIR}/model.best.pth", map_location=lambda storage, loc: storage
        )
        actor = get_network(env)
        actor.load_state_dict(state_dict)
        rewards, _ = generate_sessions(env=env, network=actor.eval(), num_sessions=100)
        env.close()
        print("mean reward:", np.mean(rewards))
