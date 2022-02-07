from pprint import pprint

from jax import grad, jit, random, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment, PickleCheckpointerCallback


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


@jit
def update(params, x, y, step_size):
    grads = grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class Experiment(IExperiment):
    def __init__(self, num_epochs: int):
        super().__init__()
        self.num_epochs = num_epochs

    def _setup_data(self):
        batch_size = 64
        self.n_targets = 10
        self.step_size = 0.01

        mnist_dataset = MNIST("./data", download=True, transform=FlattenAndCast())
        self.training_generator = NumpyLoader(
            mnist_dataset, batch_size=batch_size, num_workers=0
        )

        # Get the full train dataset (for checking accuracy while training)
        self.train_images = np.array(mnist_dataset.train_data).reshape(
            len(mnist_dataset.train_data), -1
        )
        self.train_labels = one_hot(np.array(mnist_dataset.train_labels), self.n_targets)

        # Get full test dataset
        mnist_dataset_valid = MNIST("./data", download=True, train=False)
        self.valid_images = jnp.array(
            mnist_dataset_valid.test_data.numpy().reshape(
                len(mnist_dataset_valid.test_data), -1
            ),
            dtype=jnp.float32,
        )
        self.valid_labels = one_hot(
            np.array(mnist_dataset_valid.test_labels), self.n_targets
        )

    def _setup_model(self):
        layer_sizes = [784, 512, 512, 10]
        self.params = init_network_params(layer_sizes, random.PRNGKey(0))

    def _setup_callbacks(self):
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=False,
                patience=5,
                metric_key="valid_accuracy",
                min_delta=0.01,
            ),
            "checkpointer": PickleCheckpointerCallback(
                exp_attr="params",
                logdir="./logs_jax",
                metric_key="valid_accuracy",
                minimize=False,
            ),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        self._setup_data()
        self._setup_model()
        self._setup_callbacks()

    def run_epoch(self) -> None:
        for x, y in tqdm(self.training_generator):
            y = one_hot(y, self.n_targets)
            self.params = update(self.params, x, y, self.step_size)

        train_accuracy = accuracy(self.params, self.train_images, self.train_labels)
        valid_accuracy = accuracy(self.params, self.valid_images, self.valid_labels)
        self.epoch_metrics = {
            "train_accuracy": float(train_accuracy),
            "valid_accuracy": float(valid_accuracy),
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(exp)
        pprint(self.epoch_metrics)


if __name__ == "__main__":
    Experiment(num_epochs=15).run()
