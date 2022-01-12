# Date: 12/12/2021
# Author: Sergey Kolesnikov (scitator@gmail.com)
# Licence: Apache 2.0
from typing import Any, Dict, Iterable, Tuple, Union
from collections import defaultdict
from functools import lru_cache
import os
import random


def _is_module_available(module_call):
    try:
        eval(module_call)
        return True
    except ImportError:
        return False


IS_NUMPY_AVAILABLE = _is_module_available("exec('import numpy as np')")
IS_TORCH_AVAILABLE = _is_module_available("exec('import torch')")
BATCH_METRICS = Dict[str, float]
EPOCH_METRICS = Dict[str, Union[BATCH_METRICS, float]]
EXPERIMENT_METRICS = Dict[int, EPOCH_METRICS]


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if IS_NUMPY_AVAILABLE:
        import numpy as np

        np.random.seed(seed)
    if IS_TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=42)
def _has_str_intersections(origin_string: str, strings: Tuple):
    return any(x in origin_string for x in strings)


class ICallback:
    def on_experiment_start(self, exp: "IExperiment") -> None:
        pass

    def on_epoch_start(self, exp: "IExperiment") -> None:
        pass

    def on_dataset_start(self, exp: "IExperiment") -> None:
        pass

    def on_batch_start(self, exp: "IExperiment") -> None:
        pass

    def on_batch_end(self, exp: "IExperiment") -> None:
        pass

    def on_dataset_end(self, exp: "IExperiment") -> None:
        pass

    def on_epoch_end(self, exp: "IExperiment") -> None:
        pass

    def on_experiment_end(self, exp: "IExperiment") -> None:
        pass

    def on_exception(self, exp: "IExperiment") -> None:
        pass


class IExperiment(ICallback):
    def __init__(self):
        # experiment flow: data, metrics and callbacks
        self.batch: Any = None
        self.dataset: Iterable = None
        self.dataset_key: str = None
        self.is_train_dataset: bool = False
        self.datasets: Dict[str, Iterable] = {}
        self.batch_metrics: BATCH_METRICS = defaultdict(None)
        self.dataset_metrics: EPOCH_METRICS = defaultdict(None)
        self.epoch_metrics: EPOCH_METRICS = defaultdict(None)
        self.experiment_metrics: EXPERIMENT_METRICS = defaultdict(None)
        self.callbacks: Dict[str, ICallback] = {}

        # experiment counters
        self.batch_step: int = 0
        self.dataset_batch_step: int = 0
        self.epoch_step: int = 0
        self.num_epochs: int = 1
        self.seed: int = 42

        # extras
        self.exception: Exception = None
        self.need_early_stop: bool = False

    def on_experiment_start(self, exp: "IExperiment"):
        self.batch_step: int = 0
        self.epoch_step: int = 0
        self.exception: Exception = None
        self.need_early_stop: bool = False

    def on_epoch_start(self, exp: "IExperiment"):
        self.epoch_step += 1
        self.epoch_metrics: Dict = defaultdict(None)
        set_global_seed(self.seed + self.epoch_step)

    def on_dataset_start(self, exp: "IExperiment"):
        assert self.dataset is not None, "please specify datasets `on_experiment_start`"
        self.is_train_dataset: bool = self.dataset_key.startswith("train")
        self.dataset_batch_step: int = 0
        self.dataset_metrics: Dict = defaultdict(None)
        set_global_seed(self.seed + self.epoch_step)

    def on_batch_start(self, exp: "IExperiment"):
        self.batch_step += 1
        self.dataset_batch_step += 1
        self.batch_metrics: Dict = defaultdict(None)

    def on_dataset_end(self, exp: "IExperiment"):
        self.epoch_metrics[self.dataset_key] = self.dataset_metrics.copy()

    def on_epoch_end(self, exp: "IExperiment") -> None:
        self.experiment_metrics[self.epoch_step] = self.epoch_metrics.copy()

    def on_exception(self, exp: "IExperiment"):
        raise self.exception

    def _run_event(self, event: str) -> None:
        if _has_str_intersections(event, ("_start",)):
            getattr(self, event)(self)
        for callback in self.callbacks.values():
            getattr(callback, event)(self)
        if _has_str_intersections(event, ("_end", "_exception")):
            getattr(self, event)(self)

    def run_batch(self) -> None:
        raise NotImplementedError("please implement the batch handling logic")

    def run_dataset(self) -> None:
        for self.batch in self.dataset:
            self._run_event("on_batch_start")
            self.run_batch()
            self._run_event("on_batch_end")

    def run_epoch(self) -> None:
        for self.dataset_key, self.dataset in self.datasets.items():
            self._run_event("on_dataset_start")
            self.run_dataset()
            self._run_event("on_dataset_end")

    def run_experiment(self) -> None:
        while self.epoch_step < self.num_epochs:
            if self.need_early_stop:
                break
            self._run_event("on_epoch_start")
            self.run_epoch()
            self._run_event("on_epoch_end")

    def _run(self) -> None:
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")

    def run(self) -> "IExperiment":
        try:
            self._run()
        except (Exception, KeyboardInterrupt) as ex:
            self.exception = ex
            self._run_event("on_exception")
        return self


__all__ = ["set_global_seed", "ICallback", "IExperiment"]
