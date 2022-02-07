# Date: 12/12/2021
# Author: Sergey Kolesnikov (scitator@gmail.com)
# Licence: Apache 2.0
from typing import Any, Callable, Dict, Optional, Union
import os

from accelerate import Accelerator
from accelerate.state import DistributedType
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from animus.torch import IS_TORCH_XLA_AVAILABLE

if IS_TORCH_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


def _ddp_sum_reduce(tensor: torch.Tensor) -> torch.Tensor:
    cloned = tensor.clone()
    dist.all_reduce(cloned, dist.ReduceOp.SUM)
    return cloned


def _ddp_mean_reduce(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    reduced = _ddp_sum_reduce(tensor) / world_size
    return reduced


class Engine(Accelerator):
    def spawn(self, fn: Callable, *args, **kwargs):
        return fn(*args, **kwargs)

    def setup(self, local_rank: int, world_size: int):
        pass

    def cleanup(self):
        pass

    def mean_reduce_ddp_metrics(self, metrics: Dict) -> Dict:
        if self.state.distributed_type in [
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
        ]:
            metrics = {
                k: _ddp_mean_reduce(
                    torch.tensor(v, device=self.device),
                    world_size=self.state.num_processes,
                )
                for k, v in metrics.items()
            }
        elif self.state.distributed_type == DistributedType.TPU:
            metrics = {
                k: xm.mesh_reduce(
                    k, v.item() if isinstance(v, torch.Tensor) else v, np.mean
                )
                for k, v in metrics.items()
            }
        return metrics


class CPUEngine(Engine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cpu=True, **kwargs)


class GPUEngine(Engine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cpu=False, **kwargs)


class DPEngine(Engine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cpu=False, **kwargs)

    def prepare_model(self, model):
        model = torch.nn.DataParallel(model)
        model = super().prepare_model(model)
        return model


class DDPEngine(Engine):
    def __init__(
        self,
        *args,
        address: str = "127.0.0.1",
        port: Union[str, int] = 2112,
        world_size: Optional[int] = None,
        process_group_kwargs: Dict[str, Any] = None,
        **kwargs
    ):
        self._address = os.environ.get("MASTER_ADDR", address)
        self._port = os.environ.get("MASTER_PORT", port)
        self._world_size = world_size
        self._process_group_kwargs = process_group_kwargs or {}
        self._args = args
        self._kwargs = kwargs

    def spawn(self, fn: Callable, *args, **kwargs):
        world_size: int = self._world_size or torch.cuda.device_count()
        return mp.spawn(
            fn,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def setup(self, local_rank: int, world_size: int):
        process_group_kwargs = {
            "backend": "nccl",
            "world_size": world_size,
            **self._process_group_kwargs,
        }
        os.environ["MASTER_ADDR"] = str(self._address)
        os.environ["MASTER_PORT"] = str(self._port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(**process_group_kwargs)
        super().__init__(self, *self._args, **self._kwargs)

    def cleanup(self):
        dist.destroy_process_group()

    def mean_reduce_ddp_metrics(self, metrics: Dict) -> Dict:
        metrics = {
            k: _ddp_mean_reduce(
                torch.tensor(v, device=self.device),
                world_size=self.state.num_processes,
            )
            for k, v in metrics.items()
        }
        return metrics


class XLAEngine(Engine):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def spawn(self, fn: Callable, *args, **kwargs):
        world_size: int = 8
        return xmp.spawn(fn, args=(world_size,), nprocs=world_size, start_method="fork")

    def setup(self, local_rank: int, world_size: int):
        super().__init__(self, *self._args, **self._kwargs)

    def mean_reduce_ddp_metrics(self, metrics: Dict) -> Dict:
        metrics = {
            k: xm.mesh_reduce(k, v.item() if isinstance(v, torch.Tensor) else v, np.mean)
            for k, v in metrics.items()
        }
        return metrics


__all__ = [Engine, CPUEngine, GPUEngine, DPEngine, DDPEngine, XLAEngine]
