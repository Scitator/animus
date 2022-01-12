from typing import Iterator, Optional, Sequence, Tuple
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

LOGDIR = "./logs_dqn"


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


# DQN


def get_action(env, network: nn.Module, state: np.array, epsilon: float = -1) -> int:
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state = torch.tensor(state[None], dtype=torch.float32)
        q_values = network(state).detach().cpu().numpy()[0]
        action = np.argmax(q_values)

    return int(action)


def generate_session(
    env,
    network: nn.Module,
    t_max: int = 1000,
    epsilon: float = -1,
    replay_buffer: Optional[ReplayBuffer] = None,
) -> Tuple[float, int]:
    total_reward = 0
    state = env.reset()

    for t in range(t_max):
        action = get_action(env, network, state=state, epsilon=epsilon)
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
    t_max: int = 1000,
    epsilon: float = -1,
    replay_buffer: ReplayBuffer = None,
    num_sessions: int = 100,
) -> Tuple[float, int]:
    sessions_reward, sessions_steps = 0, 0
    for i_episone in range(num_sessions):
        r, t = generate_session(
            env=env,
            network=network,
            t_max=t_max,
            epsilon=epsilon,
            replay_buffer=replay_buffer,
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


class DiscreteSamplerCallback(ICallback):
    def __init__(
        self,
        *,
        actor_attr: str,
        env,
        replay_buffer: ReplayBuffer,
        session_period: int,
        epsilon: float,
        epsilon_k: float,
        num_start_sessions: int = int(1e3),
        num_valid_sessions: int = int(1e2),
        prefix: str = "sampler",
    ):
        super().__init__()
        self.prefix = prefix
        self.env = env
        self.replay_buffer = replay_buffer
        self.session_period = session_period
        self.epsilon = epsilon
        self.epsilon_k = epsilon_k
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
            epsilon=self.epsilon,
            replay_buffer=self.replay_buffer,
            num_sessions=self.num_start_sessions,
        )
        self.actor.train()

    def on_epoch_start(self, exp: IExperiment):
        self.epsilon *= self.epsilon_k
        self.session_counter = 0
        self.session_steps = 0
        exp.epoch_metrics["sampler"] = {}

    def on_batch_end(self, exp: IExperiment):
        if exp.batch_step % self.session_period == 0:
            self.actor.eval()

            session_reward, session_steps = generate_session(
                env=self.env,
                network=self.actor,
                epsilon=self.epsilon,
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
        exp.epoch_metrics[self.prefix]["epsilon"] = self.epsilon
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
        epsilon=0.98,
        epsilon_k=0.9,
        # general
        num_epochs: int,
        env_name: str = "CartPole-v1",
    ):
        super().__init__()
        self.num_epochs = num_epochs

        self.gamma: float = gamma
        self.tau: float = tau
        self.tau_period: int = tau_period

        self.session_period = session_period
        self.epsilon = epsilon
        self.epsilon_k = epsilon_k

        self.env_name = env_name

        self.env = None
        self.replay_buffer = None
        self.network: nn.Module = None
        self.target_network: nn.Module = None

    def _setup_data(self):
        self.env = gym.make(self.env_name)

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
        self.network, self.target_network = get_network(self.env), get_network(self.env)
        set_requires_grad(self.target_network, requires_grad=False)
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.criterion = nn.MSELoss()

        lr = 3e-4
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def _setup_callbacks(self):
        # callback-based sampling is useful for exploration purposes
        # several samplers with different exploration setups could be used
        # to boost agent exploration
        self.callbacks = {
            "sampler": DiscreteSamplerCallback(
                actor_attr="network",
                env=self.env,
                replay_buffer=self.replay_buffer,
                session_period=self.session_period,
                epsilon=self.epsilon,
                epsilon_k=self.epsilon_k,
                prefix="sampler",
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="network",
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

    def handle_batch(self, batch: Sequence[np.array]) -> None:
        # model train/valid step
        states, actions, rewards, dones, next_states = batch

        # get q-values for all actions in current states
        state_qvalues = self.network(states)
        # select q-values for chosen actions
        state_action_qvalues = state_qvalues.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # compute q-values for all actions in next states
        # compute V*(next_states) using predicted next q-values
        # at the last state we shall use simplified formula:
        # Q(s,a) = r(s,a) since s' doesn't exist
        with torch.no_grad():
            next_state_qvalues = self.target_network(next_states)
            next_state_values = next_state_qvalues.max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        # compute "target q-values" for loss,
        # it's what's inside square parentheses in the above formula.
        target_state_action_qvalues = next_state_values * self.gamma + rewards

        # mean squared error loss to minimize
        loss = self.criterion(state_action_qvalues, target_state_action_qvalues.detach())
        self.batch_metrics.update({"loss": loss})
        self.dataset_metrics["loss"].append(loss.item())

        if self.is_train_dataset:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.batch_step % self.tau_period == 0:
                soft_update(self.target_network, self.network, self.tau)

    def on_dataset_end(self, exp: "IExperiment"):
        self.dataset_metrics["loss"] = np.mean(self.dataset_metrics["loss"])
        super().on_dataset_end(exp)

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    # exp settings, ~training
    gamma = 0.99
    tau = 0.01
    tau_period = 1  # in batches
    # callback, ~exploration
    session_period = 100  # in batches
    epsilon = 0.98
    epsilon_k = 0.9
    # extras
    env_name = "CartPole-v1"

    # train
    exp = Experiment(
        num_epochs=10,
        gamma=gamma,
        tau=tau,
        tau_period=tau_period,
        session_period=session_period,
        epsilon=epsilon,
        epsilon_k=epsilon_k,
        env_name=env_name,
    ).run()

    # evaluate
    env = gym.wrappers.Monitor(gym.make(env_name), directory="videos_dqn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        f"{LOGDIR}/network.best.pth", map_location=lambda storage, loc: storage
    )
    network = get_network(env)
    network.load_state_dict(state_dict)
    rewards, _ = generate_sessions(env=env, network=network.eval(), num_sessions=100)
    env.close()
    print("mean reward:", np.mean(rewards))
