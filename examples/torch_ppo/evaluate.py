import argparse
import os
import time

import gym
import torch

from animus import set_global_seed

from src.acmodel import ACModel
from src.agent import Agent
from src.settings import LOGDIR
from src.utils import ParallelEnv, synthesize

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="name of the environment (REQUIRED)"
    )
    # parser.add_argument(
    #     "--model", required=True, help="name of the trained model (REQUIRED)"
    # )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of episodes of evaluation (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--procs", type=int, default=16, help="number of processes (default: 16)"
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        default=False,
        help="action with highest probability is selected",
    )
    parser.add_argument(
        "--worst-episodes-to-show",
        type=int,
        default=10,
        help="how many worst episodes to show",
    )
    parser.add_argument(
        "--recurrent", action="store_true", default=False, help="add a LSTM to the model"
    )
    args = parser.parse_args()

    # Set seed for all randomness sources
    set_global_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        env = gym.make(args.env)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent
    acmodel = ACModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        recurrent=args.recurrent,
    )
    checkpoint = torch.load(
        os.path.join(LOGDIR, "acmodel.best.pth"),
        map_location=lambda storage, loc: storage,
    )
    acmodel.load_state_dict(checkpoint)
    agent = Agent(
        acmodel=acmodel,
        device=device,
        argmax=args.argmax,
    )
    print("Agent loaded\n")

    # Initialize logs
    logs = {"num_steps_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()
    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_steps = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_steps += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_steps_per_episode"].append(log_episode_num_steps[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_steps *= mask

    end_time = time.time()

    # Print logs
    num_steps = sum(logs["num_steps_per_episode"])
    fps = num_steps / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = synthesize(logs["return_per_episode"])
    num_steps_per_episode = synthesize(logs["num_steps_per_episode"])

    print(
        "S {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}".format(
            num_steps,
            fps,
            duration,
            *return_per_episode.values(),
            *num_steps_per_episode.values(),
        )
    )

    # Print worst episodes
    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(
            range(len(logs["return_per_episode"])),
            key=lambda k: logs["return_per_episode"][k],
        )
        for i in indexes[:n]:
            print(
                "- episode {}: R={}, F={}".format(
                    i, logs["return_per_episode"][i], logs["num_steps_per_episode"][i]
                )
            )
