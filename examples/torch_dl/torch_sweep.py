from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import wandb

from animus import IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback

HPARAMS = dict(
    num_hidden1=16,
    num_hidden2=16,
    learning_rate=0.02,
    batch_size=100,
    num_epochs=5,
)


class Experiment(IExperiment):
    def _setup_hparams(self):
        wandb.init(config=HPARAMS, project="animus-torch-sweep")
        self.hparams = wandb.config
        self.num_epochs = self.hparams.num_epochs

    def _setup_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_data = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        valid_data = datasets.MNIST("./data", train=False, transform=transform)
        kwargs = dict(
            batch_size=self.hparams.batch_size,
            num_workers=1,
            pin_memory=torch.cuda.is_available(),
        )
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, **kwargs)
        self.datasets = {"train": train_loader, "valid": valid_loader}

    def _setup_model(self):
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, self.hparams.num_hidden1),
            nn.ReLU(),
            nn.Linear(self.hparams.num_hidden1, self.hparams.num_hidden2),
            nn.ReLU(),
            nn.Linear(self.hparams.num_hidden2, 10),
        ).to(self.device)
        wandb.watch(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate
        )

    def _setup_callbacks(self):
        self.callbacks = {
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="model",
                logdir="./logs_torch_sweep",
                dataset_key="valid",
                metric_key="accuracy",
                minimize=False,
            ),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_hparams()
        self._setup_data()
        self._setup_model()
        self._setup_callbacks()

    def run_dataset(self) -> None:
        total_loss, total_accuracy = 0.0, 0.0

        self.model.train(self.is_train_dataset)
        with torch.set_grad_enabled(self.is_train_dataset):
            for data, target in tqdm(self.dataset):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                total_loss += loss.sum().item()
                total_accuracy += pred.eq(target.view_as(pred)).sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= len(self.dataset)
        total_accuracy /= len(self.dataset) * self.hparams.batch_size
        self.dataset_metrics = {"loss": total_loss, "accuracy": total_accuracy}

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)
        flatten_metrics = {
            f"{dataset}_{metric}": value
            for dataset, v in self.epoch_metrics.items()
            for metric, value in v.items()
        }
        wandb.log(flatten_metrics)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        wandb.save("./logs_torch_sweep/*")
        wandb.save("./logs_torch_sweep/*")


if __name__ == "__main__":
    Experiment().run()
