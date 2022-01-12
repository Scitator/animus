# Date: 12/12/2021
# Author: Sergey Kolesnikov (scitator@gmail.com)
# Licence: Apache 2.0
from typing import Any, List
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
import json
import os
import pickle
import shutil

from animus.core import ICallback, IExperiment


def _is_better_min(score, best, min_delta):
    return score <= (best - min_delta)


def _is_better_max(score, best, min_delta):
    return score >= (best + min_delta)


class EarlyStoppingCallback(ICallback):
    def __init__(
        self,
        minimize: bool,
        patience: int,
        metric_key: str,
        dataset_key: str = None,
        min_delta: float = 1e-6,
    ) -> None:
        super().__init__()
        self._dataset_key = dataset_key
        self._metric_key = metric_key
        self._patience = patience
        self._patience_step = 0
        self.best_score = None
        self.best_epoch = 0
        if minimize:
            self._is_better = partial(_is_better_min, min_delta=min_delta)
        else:
            self._is_better = partial(_is_better_max, min_delta=min_delta)

    def on_experiment_start(self, exp: "IExperiment") -> None:
        self._patience_step = 0
        self.best_score = None

    def on_epoch_end(self, exp: "IExperiment") -> None:
        if self._dataset_key is not None:
            score = exp.epoch_metrics[self._dataset_key][self._metric_key]
        else:
            score = exp.epoch_metrics[self._metric_key]

        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = exp.epoch_step
            self._patience_step = 0
        else:
            self._patience_step += 1
            if self._patience_step >= self._patience:
                exp.need_early_stop = True


Checkpoint = namedtuple("Checkpoint", field_names=["obj", "logpath", "metric"])


class ICheckpointerCallback(ABC, ICallback):
    def __init__(
        self,
        exp_attr: str,
        logdir: str,
        topk: int = 1,
        dataset_key: str = None,
        metric_key: str = None,
        minimize: bool = None,
    ) -> None:
        super().__init__()
        self.logdir = logdir
        self._exp_attr = exp_attr
        self._topk = topk
        self._dataset_key = dataset_key
        self._metric_key = metric_key
        if minimize is not None:
            assert metric_key is not None, "please define the metric to track"
            self._minimize = minimize
            self.on_epoch_end = self.on_epoch_end_best
        else:
            self._minimize = False
            self.on_epoch_end = self.on_epoch_end_last
        os.makedirs(self.logdir, exist_ok=True)
        self._storage: List[Checkpoint] = []

    @abstractmethod
    def save(self, exp: IExperiment, obj: Any, logprefix: str) -> str:
        pass

    def _handle_epoch(self, exp: IExperiment, score: float):
        obj = exp.__dict__[self._exp_attr]
        logprefix = f"{self.logdir}/{self._exp_attr}.{exp.epoch_step:03d}"
        logpath = self.save(exp, obj, logprefix)
        self._storage.append(Checkpoint(obj=obj, logpath=logpath, metric=score))
        self._storage = sorted(
            self._storage, key=lambda x: x.metric, reverse=not self._minimize
        )
        if len(self._storage) > self._topk:
            last_item = self._storage.pop(-1)
            if os.path.isfile(last_item.logpath):
                try:
                    os.remove(last_item.logpath)
                except OSError:
                    pass
            elif os.path.isdir(last_item.logpath):
                shutil.rmtree(last_item.logpath, ignore_errors=True)
        with open(f"{self.logdir}/{self._exp_attr}.storage.json", "w") as fout:
            stats = {
                "exp_attr": self._exp_attr,
                "logdir": self.logdir,
                "topk": self._topk,
                "dataset_key": self._dataset_key,
                "metric_key": self._metric_key,
                "minimize": self._minimize,
            }
            storage = [{"logpath": x.logpath, "metric": x.metric} for x in self._storage]
            stats["storage"] = storage
            json.dump(stats, fout, indent=2, ensure_ascii=False)

    def on_experiment_start(self, exp: "IExperiment") -> None:
        self._storage = []

    def on_epoch_end_best(self, exp: "IExperiment") -> None:
        if self._dataset_key is not None:
            score = exp.epoch_metrics[self._dataset_key][self._metric_key]
        else:
            score = exp.epoch_metrics[self._metric_key]
        self._handle_epoch(exp=exp, score=score)

        best_logprefix = f"{self.logdir}/{self._exp_attr}.best"
        self.save(exp, self._storage[0].obj, best_logprefix)

    def on_epoch_end_last(self, exp: "IExperiment") -> None:
        self._handle_epoch(exp=exp, score=exp.epoch_step)


class PickleCheckpointerCallback(ICheckpointerCallback):
    def save(self, exp: IExperiment, obj: Any, logprefix: str) -> str:
        logpath = f"{logprefix}.pkl"
        with open(logpath, "wb") as fout:
            pickle.dump(obj, fout)
        return logpath


__all__ = [EarlyStoppingCallback, ICheckpointerCallback, PickleCheckpointerCallback]
