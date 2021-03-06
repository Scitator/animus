from argparse import ArgumentParser, RawTextHelpFormatter
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import EngineCheckpointerCallback
from animus.torch.engine import CPUEngine, DDPEngine, DPEngine, GPUEngine, XLAEngine

E2E = {
    "cpu": CPUEngine,
    "gpu": GPUEngine,
    "dp": DPEngine,
    "ddp": DDPEngine,
    "xla": XLAEngine,
}


class Experiment(IExperiment):
    def __init__(self, num_epochs: int, engine):
        super().__init__()
        self.num_epochs = num_epochs
        self.engine = engine

    def _setup_data(self):
        self.batch_size = 64
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_data = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        valid_data = datasets.MNIST("./data", train=False, transform=transform)
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=1,
            # pin_memory=torch.cuda.is_available(),
        )
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, **kwargs)
        train_loader, valid_loader = self.engine.prepare(train_loader, valid_loader)
        self.datasets = {"train": train_loader, "valid": valid_loader}

    def _setup_model(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        ).to(self.engine.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.model, self.optimizer = self.engine.prepare(self.model, self.optimizer)

    def _setup_callbacks(self):
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=False,
                patience=5,
                dataset_key="valid",
                metric_key="accuracy",
                min_delta=0.01,
            ),
            "checkpointer": EngineCheckpointerCallback(
                exp_attr="model",
                logdir="./logs_torch_dl",
                dataset_key="valid",
                metric_key="accuracy",
                minimize=False,
            ),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self.engine.setup(self._local_rank, self._world_size)
        with self.engine.local_main_process_first():
            self._setup_data()
        self._setup_model()
        self._setup_callbacks()

    def run_dataset(self) -> None:
        total_loss, total_accuracy = 0.0, 0.0

        self.model.train(self.is_train_dataset)
        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(
                tqdm(self.dataset, disable=not self.engine.is_local_main_process)
            ):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                total_loss += loss.sum().item()
                total_accuracy += pred.eq(target.view_as(pred)).sum().item()
                if self.is_train_dataset:
                    self.engine.backward(loss)
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step
        total_accuracy /= self.dataset_batch_step * self.batch_size
        self.dataset_metrics = {"loss": total_loss, "accuracy": total_accuracy}
        self.dataset_metrics = self.engine.mean_reduce_ddp_metrics(self.dataset_metrics)
        self.dataset_metrics = {k: float(v) for k, v in self.dataset_metrics.items()}

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        if self.engine.is_local_main_process:
            pprint(self.epoch_metrics)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self.engine.cleanup()

    def _run_local(self, local_rank: int = -1, world_size: int = 1) -> None:
        self._local_rank, self._world_size = local_rank, world_size
        self._run_event("on_experiment_start")
        self.run_experiment()
        self._run_event("on_experiment_end")

    def _run(self) -> None:
        self.engine.spawn(self._run_local)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--engine", type=str, choices=list(E2E.keys()), required=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    args, unknown_args = parser.parse_known_args()
    Experiment(num_epochs=15, engine=E2E[args.engine](fp16=args.fp16)).run()
