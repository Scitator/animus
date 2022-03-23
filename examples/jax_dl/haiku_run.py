from typing import Mapping, Tuple
from pprint import pprint

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment, PickleCheckpointerCallback

Batch = Mapping[str, np.ndarray]


def bcast_local_devices(value):
    """Broadcasts an object to all local devices."""
    devices = jax.local_devices()
    return jax.tree_map(
        lambda v: jax.device_put_sharded(len(devices) * [v], devices), value
    )


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
        return {"images": images, "labels": labels}


Batch = Mapping[str, np.ndarray]


class Experiment(IExperiment):
    def __init__(self, num_epochs: int):
        super().__init__()
        self.num_epochs = num_epochs

    def _forward_fn(self, batch: Batch) -> jnp.ndarray:
        hk.set_state("counter", jnp.zeros([]))  # just a trick for state initialization
        x = batch["images"]
        net = hk.Sequential(
            [
                hk.Flatten(),
                hk.Linear(300),
                jax.nn.relu,
                hk.Linear(100),
                jax.nn.relu,
                hk.Linear(10),
            ]
        )
        return net(x)

    def _loss_fn(
        self, params: hk.Params, state: hk.State, rng: jnp.ndarray, batch: Batch
    ) -> Tuple[jnp.ndarray, hk.State]:
        logits, state = self.forward.apply(params, state, rng, batch)
        labels = jax.nn.one_hot(batch["labels"], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return (softmax_xent + 1e-4 * l2_loss, state)

    def _eval_batch(
        self, params: hk.Params, state: hk.State, rng: jnp.ndarray, batch: Batch
    ) -> jnp.ndarray:
        logits, _ = self.forward.apply(params, state, rng, batch)
        y = batch["labels"]
        return jnp.sum(jnp.argmax(logits, axis=-1) == y)

    def _update_fn(
        self,
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        rng: jnp.ndarray,
        batch: Batch,
    ) -> Tuple[hk.Params, hk.State, optax.OptState]:
        grad_loss_fn = jax.grad(self._loss_fn, has_aux=True)
        grads, state = grad_loss_fn(params, state, rng, batch)

        updates, opt_state = self._optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, state, opt_state

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
        kwargs = dict(batch_size=self.batch_size, num_workers=0, pin_memory=False,)
        collate_fn = CustomCollate(len(jax.devices()))
        train_loader = DataLoader(
            train_data, shuffle=True, collate_fn=collate_fn, **kwargs
        )
        valid_loader = DataLoader(
            valid_data, shuffle=False, collate_fn=collate_fn, **kwargs
        )
        self.datasets = {"train": train_loader, "valid": valid_loader}

    def _setup_model(self):
        rng = bcast_local_devices(jax.random.PRNGKey(42))

        self.forward = hk.transform_with_state(self._forward_fn)

        self._update_fn = jax.pmap(
            self._update_fn, axis_name="i", donate_argnums=(0, 1, 2)
        )
        self._eval_batch = jax.pmap(
            self._eval_batch, axis_name="i", donate_argnums=(0, 1)
        )
        self._update_fn = jax.jit(self._update_fn)
        self._eval_batch = jax.jit(self._eval_batch)

        self._optimizer = optax.adam(1e-3)

        # Initialize network and optimiser; note we draw an input to get shapes.
        batch = next(iter(self.datasets["train"]))

        init_net = jax.pmap(self.forward.init)
        init_opt = jax.pmap(self._optimizer.init)

        self._params, self._state = init_net(rng, batch)
        self._opt_state = init_opt(self._params)

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
                logdir="./logs_haiku",
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
        rng = bcast_local_devices(jax.random.PRNGKey(self.epoch_step))
        total_accuracy = 0.0
        for batch in tqdm(self.dataset):
            if self.is_train_dataset:
                self._params, self._state, self._opt_state = self._update_fn(
                    self._params, self._state, self._opt_state, rng, batch
                )
            total_accuracy += self._eval_batch(self._params, self._state, rng, batch)

        total_accuracy = jnp.sum(total_accuracy)
        total_accuracy /= len(self.dataset) * self.batch_size
        self.dataset_metrics = {"accuracy": float(total_accuracy)}

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    Experiment(num_epochs=15).run()
