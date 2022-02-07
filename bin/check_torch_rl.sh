#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# pip install animus
# pip install gym numpy torch tqdm

CUDA_VISIBLE_DEVICES="" python examples/torch_rl/torch_dqn.py
CUDA_VISIBLE_DEVICES="" python examples/torch_rl/torch_ddpg.py
CUDA_VISIBLE_DEVICES="" python examples/torch_rl/torch_reinforce.py

export CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")

if [[ $CUDA -ne 0 ]]; then
    CUDA_VISIBLE_DEVICES="0" python examples/torch_rl/torch_dqn.py
    CUDA_VISIBLE_DEVICES="0" python examples/torch_rl/torch_ddpg.py
    CUDA_VISIBLE_DEVICES="0" python examples/torch_rl/torch_reinforce.py
fi
