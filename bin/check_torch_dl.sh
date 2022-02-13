#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# pip install animus
# pip install accelerate packaging torch torchvision tqdm

python examples/torch_dl/torch_run.py --engine="cpu"

export CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
export NGPU=$(python -c "import torch; print(int(torch.cuda.device_count() >= 2))")
export TXLA=$(python -c """
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    print(1)
except ImportError:
    print(0)
""")

if [[ $CUDA -ne 0 ]]; then
    python examples/torch_dl/torch_run.py --engine="gpu"
    python examples/torch_dl/torch_run.py --engine="gpu" --fp16
fi

if [[ $CUDA -ne 0 ]] && [[ $NGPU -ne 0 ]]; then
    python examples/torch_dl/torch_run.py --engine="dp"
    python examples/torch_dl/torch_run.py --engine="dp" --fp16
    python examples/torch_dl/torch_run.py --engine="ddp"
    python examples/torch_dl/torch_run.py --engine="ddp" --fp16
fi

if [[ $TXLA -ne 0 ]]; then
    python examples/torch_dl/torch_run.py --engine="xla"
fi
