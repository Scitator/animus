#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# pip install animus
# pip install "jax[cpu]"
# pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install haiku numpy optax torch torchvision tqdm

CUDA_VISIBLE_DEVICES="" python examples/jax_dl/jax_run.py
CUDA_VISIBLE_DEVICES="" python examples/jax_dl/haiku_run.py

export CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
export NGPU=$(python -c "import torch; print(int(torch.cuda.device_count() >= 2))")

if [[ $CUDA -ne 0 ]]; then
    CUDA_VISIBLE_DEVICES="0" python examples/jax_dl/jax_run.py
    CUDA_VISIBLE_DEVICES="0" python examples/jax_dl/haiku_run.py
fi

if [[ $CUDA -ne 0 ]] && [[ $NGPU -ne 0 ]]; then
    CUDA_VISIBLE_DEVICES="0,1" python examples/jax_dl/jax_run.py
    CUDA_VISIBLE_DEVICES="0,1" python examples/jax_dl/haiku_run.py
fi
