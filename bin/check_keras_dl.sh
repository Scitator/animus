#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# pip install animus
# pip install numpy tensorflow torch torchvision tqdm

CUDA_VISIBLE_DEVICES="" python examples/keras_dl/keras_run_tf.py
CUDA_VISIBLE_DEVICES="" python examples/keras_dl/keras_run_pt.py

export CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
export NGPU=$(python -c "import torch; print(int(torch.cuda.device_count() >= 2))")

if [[ $CUDA -ne 0 ]]; then
    CUDA_VISIBLE_DEVICES="0" python examples/keras_dl/keras_run_tf.py
    CUDA_VISIBLE_DEVICES="0" python examples/keras_dl/keras_run_pt.py
fi

if [[ $CUDA -ne 0 ]] && [[ $NGPU -ne 0 ]]; then
    CUDA_VISIBLE_DEVICES="0,1" python examples/keras_dl/keras_run_tf.py
    CUDA_VISIBLE_DEVICES="0,1" python examples/keras_dl/keras_run_pt.py
fi
