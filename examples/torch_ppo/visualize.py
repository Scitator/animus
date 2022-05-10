import argparse
import os

import gym
import numpy as np
import torch

from animus import set_global_seed

from src.acmodel import ACModel
from src.agent import Agent
from src.settings import LOGDIR

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run (REQUIRED)"
    )
    # parser.add_argument(
    #     "--model", required=True, help="name of the trained model (REQUIRED)"
    # )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--shift",
        type=int,
        default=0,
        help="number of times the environment is reset at the beginning (default: 0)",
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        default=False,
        help="select the action with highest probability (default: False)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.1,
        help="pause duration between two consequent actions of the agent (default: 0.1)",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="store output as gif with the given filename",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="number of episodes to visualize"
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

    # Load environment

    env = gym.make(args.env)
    for _ in range(args.shift):
        env.reset()
    print("Environment loaded\n")

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

    # Run the agent

    if args.gif:
        from array2gif import write_gif

        frames = []

    # Create a window to view the environment
    # env.render("human")

    for episode in range(args.episodes):
        obs = env.reset()

        while True:
            env.render("human")
            if args.gif:
                frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))

            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            if done:  #  or env.window.closed:
                break

        # if env.window.closed:
        #     break

    if args.gif:
        print("Saving gif... ", end="")
        write_gif(np.array(frames), args.gif + ".gif", fps=1 / args.pause)
        print("Done.")
