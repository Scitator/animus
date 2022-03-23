from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment
from animus.keras.callbacks import KerasCheckpointerCallback


class CustomCollate:
    def __call__(self, batch):
        images = np.vstack([x[0][None] for x in batch])
        labels = np.vstack([x[1] for x in batch])[:, 0]
        return images, labels


class Experiment(IExperiment):
    def __init__(self, num_epochs: int):
        super().__init__()
        self.num_epochs = num_epochs

    def _setup_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                lambda x: np.array(x),
            ]
        )
        target_transform = lambda y: int(y)
        train_data = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform,
        )
        valid_data = datasets.MNIST(
            "./data",
            train=False,
            transform=transform,
            target_transform=target_transform,
        )
        kwargs = dict(batch_size=64, num_workers=0, pin_memory=False,)
        collate_fn = CustomCollate()
        train_loader = DataLoader(
            train_data, shuffle=True, collate_fn=collate_fn, **kwargs
        )
        valid_loader = DataLoader(
            valid_data, shuffle=False, collate_fn=collate_fn, **kwargs
        )
        self.datasets = {"train": train_loader, "valid": valid_loader}

    def _setup_model(self):
        self.model = Sequential(
            layers=[
                layers.Flatten(),
                layers.Dense(300, activation="relu"),
                layers.Dense(100, activation="relu"),
                layers.Dense(10),
            ]
        )
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

    def _setup_callbacks(self):
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=False,
                patience=5,
                dataset_key="valid",
                metric_key="accuracy",
                min_delta=0.01,
            ),
            "checkpointer": KerasCheckpointerCallback(
                exp_attr="model",
                logdir="./logs_keras",
                dataset_key="valid",
                metric_key="accuracy",
                minimize=False,
            ),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self._setup_data()
        self._setup_model()
        self._setup_callbacks()

    @tf.function
    def train_step(self, images, labels, loss_metric, accuracy_metric):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.criterion(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss_metric(loss)
        accuracy_metric(labels, predictions)

    @tf.function
    def valid_step(self, images, labels, loss_metric, accuracy_metric):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.criterion(labels, predictions)

        loss_metric(t_loss)
        accuracy_metric(labels, predictions)

    def run_dataset(self) -> None:
        loss_metric = tf.keras.metrics.Mean(name=f"{self.dataset_key}_loss")
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
            name=f"{self.dataset_key}_accuracy"
        )

        if self.is_train_dataset:
            for images, labels in tqdm(self.dataset):
                self.train_step(images, labels, loss_metric, accuracy_metric)
        else:
            for images, labels in tqdm(self.dataset):
                self.valid_step(images, labels, loss_metric, accuracy_metric)

        self.dataset_metrics = {
            "loss": float(loss_metric.result().numpy()),
            "accuracy": float(accuracy_metric.result().numpy()),
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    Experiment(num_epochs=15).run()
