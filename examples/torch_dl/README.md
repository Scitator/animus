## Tl;dr
```bash
pip install accelerate animus torch torchvision tqdm
python torch_run.py --accelerator="cpu"
python torch_run.py --accelerator="gpu"
python torch_run.py --accelerator="ddp"

pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install accelerate animus tqdm
python torch_run.py --accelerator="xla"


# wandb sweep https://docs.wandb.ai/sweeps
pip install accelerate animus torch torchvision tqdm wandb
wandb sweep sweep.yaml
wandb agent <USERNAME/PROJECTNAME/SWEEPID>
```
