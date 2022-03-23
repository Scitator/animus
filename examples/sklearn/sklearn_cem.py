import pickle
from pprint import pprint

import gym
import numpy as np
from sklearn.neural_network import MLPClassifier

from animus import EarlyStoppingCallback, IExperiment, PickleCheckpointerCallback

LOGDIR = "./logs_cem"


def generate_session(env, agent, t_max=1000):
    n_actions = env.action_space.n
    states, actions = [], []
    total_reward = 0

    s = env.reset()
    for _ in range(t_max):
        # predict array of action probabilities
        probs = agent.predict_proba([s])[0]
        # sample action with such probabilities
        a = np.random.choice(n_actions, p=probs)
        # make a step
        new_s, r, done, info = env.step(a)
        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r
        # update state
        s = new_s
        if done:
            break

    return states, actions, total_reward


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions,
        both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below.
    Please don't assume that states are integers (they'll get different later).
    """

    # Compute minimum reward for elite sessions. Hint: use np.percentile
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = [
        x for i, x in enumerate(states_batch) if rewards_batch[i] >= reward_threshold
    ]
    elite_actions = [
        x for i, x in enumerate(actions_batch) if rewards_batch[i] >= reward_threshold
    ]
    elite_states = [x for y in elite_states for x in y]
    elite_actions = [x for y in elite_actions for x in y]

    return elite_states, elite_actions


class Experiment(IExperiment):
    def __init__(
        self,
        *,
        # trainer settings, ~training
        num_sessions: int,
        percentile: float,
        # general
        num_epochs: int,
        env_name: str = "CartPole-v1",
        reward_threshold: float = np.inf,
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.env_name = env_name
        self.num_sessions = num_sessions
        self.percentile = percentile
        self.reward_threshold = reward_threshold

        self.env = None
        self.agent = None

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        # setup data
        self.env = gym.make("CartPole-v0")
        self.n_actions = self.env.action_space.n
        # setup model
        self.agent = MLPClassifier(
            hidden_layer_sizes=(20, 20),
            activation="tanh",
            warm_start=True,  # keep progress between .fit(...) calls
            max_iter=1,  # make only 1 iteration on each .fit(...)
        )
        # initialize agent to the dimension of state an amount of actions
        self.agent.fit([self.env.reset()] * self.n_actions, range(self.n_actions))
        # early-stop setup
        if not np.isfinite(self.reward_threshold):
            self.callbacks = {
                "early-stop": EarlyStoppingCallback(
                    minimize=False, patience=5, metric_key="reward", min_delta=1.0
                ),
                "checkpointer": PickleCheckpointerCallback(
                    exp_attr="agent", logdir=LOGDIR, metric_key="reward", minimize=False,
                ),
            }

    def run_epoch(self) -> None:
        # generate new sessions
        sessions = [
            generate_session(env=self.env, agent=self.agent)
            for _ in range(self.num_sessions)
        ]
        states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))
        # select elite actions
        elite_states, elite_actions = select_elites(
            states_batch, actions_batch, rewards_batch, percentile=self.percentile
        )
        try:
            self.agent.partial_fit(elite_states, elite_actions)
        except Exception:
            # hack
            addition = np.array([self.env.reset()] * self.n_actions)
            elite_states = np.vstack((elite_states, addition))
            elite_actions = np.hstack((elite_actions, list(range(self.n_actions))))
            self.agent.partial_fit(elite_states, elite_actions)

        epoch_reward = np.mean(rewards_batch)
        self.epoch_metrics["reward"] = epoch_reward

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    # exp settings, ~training
    num_sessions = 100
    percentile = 70
    # extras
    env_name = "CartPole-v1"

    # train
    Experiment(
        num_epochs=100,
        num_sessions=num_sessions,
        percentile=percentile,
        # reward_threshold=190,
    ).run()

    # evaluate
    try:
        env = gym.wrappers.Monitor(
            gym.make(env_name), directory="videos_cem", force=True
        )
        with open(f"{LOGDIR}/agent.best.pkl", "rb") as fin:
            agent = pickle.load(fin)
        sessions = [generate_session(env=env, agent=agent) for _ in range(100)]
        env.close()
        _, _, rewards = map(np.array, zip(*sessions))
        print("mean reward:", np.mean(rewards))
    except Exception:
        env = gym.make(env_name)
        with open(f"{LOGDIR}/agent.best.pkl", "rb") as fin:
            agent = pickle.load(fin)
        sessions = [generate_session(env=env, agent=agent) for _ in range(100)]
        env.close()
        _, _, rewards = map(np.array, zip(*sessions))
        print("mean reward:", np.mean(rewards))
