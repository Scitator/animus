#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# pip install animus
# pip install gym numpy torch tqdm

CUDA_VISIBLE_DEVICES="" python examples/torch_rl/torch_dqn.py
CUDA_VISIBLE_DEVICES="" python examples/torch_rl/torch_ddpg.py
CUDA_VISIBLE_DEVICES="" python examples/torch_rl/torch_reinforce.py
