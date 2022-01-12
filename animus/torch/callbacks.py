# Date: 12/12/2021
# Author: Sergey Kolesnikov (scitator@gmail.com)
# Licence: Apache 2.0
from typing import Any

import torch
from torch import nn

from animus.callbacks import ICheckpointerCallback
from animus.core import IExperiment
from animus.torch import IS_ACCELERATE_AVAILABLE

if IS_ACCELERATE_AVAILABLE:
    from accelerate import Accelerator


class TorchCheckpointerCallback(ICheckpointerCallback):
    def save(self, exp: IExperiment, obj: Any, logprefix: str) -> str:
        if isinstance(obj, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            obj = obj.module
        logpath = f"{logprefix}.pth"
        if isinstance(obj, nn.Module):
            torch.save(obj.state_dict(), logpath)
        else:
            torch.save(obj, logpath)
        return logpath


class AccelerateCheckpointerCallback(ICheckpointerCallback):
    def on_experiment_start(self, exp: "IExperiment") -> None:
        assert isinstance(getattr(exp, "accelerator", None), Accelerator)

    def save(self, exp: IExperiment, obj: Any, logprefix: str) -> str:
        logpath = f"{logprefix}.pth"
        if isinstance(obj, nn.Module):
            exp.accelerator.wait_for_everyone()
            obj = exp.accelerator.unwrap_model(obj)
            exp.accelerator.save(obj.state_dict(), logpath)
        else:
            exp.accelerator.save(obj, logpath)
        return logpath


__all__ = [TorchCheckpointerCallback, AccelerateCheckpointerCallback]
