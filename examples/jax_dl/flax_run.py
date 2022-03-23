from typing import Mapping
from pprint import pprint

from flax import linen as nn
from flax.training import train_state
import jax
from jax import lax
import jax.lib.xla_bridge as xb
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment, PickleCheckpointerCallback

Batch = Mapping[str, np.ndarray]
NUM_CLASSES = 10


def _pmap_device_order():
    # match the default device assignments used in pmap:
    # for single-host, that's the XLA default device assignment
    # for multi-host, it's the order of jax.local_devices()
    if jax.process_count() == 1:
        return [
            d
            for d in xb.get_backend().get_default_device_assignment(jax.device_count())
            if d.process_index == jax.process_index()
        ]
    else:
        return jax.local_devices()


def replicate(tree, devices=None):
    devices = devices or _pmap_device_order()
    return jax.device_put_replicated(tree, devices)


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    x = labels[..., None] == jnp.arange(num_classes).reshape((1,) * labels.ndim + (-1,))
    x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
    one_hot_labels = onehot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def classification_eval(*, logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


class CustomCollate:
    def __init__(self, n_devices: int) -> None:
        self.n_devices = n_devices

    @staticmethod
    def split(arr, n_devices):
        """Splits the first axis of `arr` evenly across the number of devices."""
        assert len(arr) % n_devices == 0
        return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

    def __call__(self, batch):
        images = np.vstack([x[0][None] for x in batch])
        labels = np.vstack([x[1] for x in batch])[:, 0]
        images = self.split(images, self.n_devices)
        labels = self.split(labels, self.n_devices)
        return {"image": images, "label": labels}


class MnistCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=300)(x)
        x = nn.relu(x)
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        x = nn.Dense(features=NUM_CLASSES)(x)
        return x


class Experiment(IExperiment):
    def __init__(self, num_epochs: int):
        super().__init__()
        self.num_epochs = num_epochs
        self._state = None

    def _setup_data(self):
        self.batch_size = 64
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
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
        )
        collate_fn = CustomCollate(len(jax.devices()))
        train_loader = DataLoader(
            train_data, shuffle=True, collate_fn=collate_fn, **kwargs
        )
        valid_loader = DataLoader(
            valid_data, shuffle=False, collate_fn=collate_fn, **kwargs
        )
        self.datasets = {"train": train_loader, "valid": valid_loader}

    def _init_train_state(self, rng: jnp.ndarray, learning_rate: float, momentum: float):
        """Creates initial `TrainState`."""
        model = MnistCNN()
        cnn_init = jax.jit(model.init)
        params = cnn_init({"params": rng}, jnp.ones([1, 28, 28, 1]))["params"]
        tx = optax.sgd(learning_rate, momentum)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        return state

    def _loss_fn(self, params, batch: Batch):
        logits = self._state.apply_fn({"params": params}, batch["image"])
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, logits

    def _train_step(self, state: train_state.TrainState, batch: Batch):
        """Train for a single step."""
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params, batch)
        grads = lax.pmean(grads, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        return state

    def _eval_step(self, state: train_state.TrainState, batch: Batch):
        logits = state.apply_fn({"params": state.params}, batch["image"])
        return classification_eval(logits=logits, labels=batch["label"])

    def _setup_model(self):
        init_rng = jax.random.PRNGKey(42)
        learning_rate, momentum = 0.1, 0.9
        state = self._init_train_state(init_rng, learning_rate, momentum)
        self._state = replicate(state)
        self.train_step = jax.pmap(self._train_step, axis_name="batch")
        self.eval_step = jax.pmap(self._eval_step, axis_name="batch")
        del init_rng

    def _setup_callbacks(self):
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=False,
                patience=5,
                dataset_key="valid",
                metric_key="accuracy",
                min_delta=0.01,
            ),
            "checkpointer": PickleCheckpointerCallback(
                exp_attr="_params",
                logdir="./logs_flax",
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

    def run_dataset(self) -> None:
        total_accuracy = 0.0
        for batch in tqdm(self.dataset):
            if self.is_train_dataset:
                self._state = self.train_step(self._state, batch)
            metrics = self.eval_step(self._state, batch)
            total_accuracy += metrics["accuracy"]

        total_accuracy = jnp.sum(total_accuracy)
        total_accuracy /= len(self.dataset) * self.batch_size
        self.dataset_metrics = {"accuracy": float(total_accuracy)}
        self._params = self._state.params

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    Experiment(num_epochs=15).run()
