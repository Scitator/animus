# Animus

> One framework to rule them all.

Animus is a "write it yourself"-based machine learning framework.<br/>
Please see `examples/` for more information.<br/>
Framework architecture is mainly inspired by [Catalyst](https://github.com/catalyst-team/catalyst).


### FAQ

<details>
<summary>What is Animus?</summary>
<p>

Animus is a general-purpose for-loop-based experiment wrapper. It divides ML experiment with the straightforward logic:
```python
def run(experiment):
    for epoch in experiment.epochs:
        for dataset in epoch.datasets:
            for batch in dataset.batches:
                handle_batch(batch)
```
Each for encapsulated with `on_{for}_start`, `run_{for}`, and `on_{for}_end` for  customisation purposes. Moreover, each for has its own metrics storage: `{for}_metrics` (`batch_metrics`, `dataset_metrics`, `epoch_metrics`, `experiment_metrics`).

</p>
</details>


<details>
<summary>What are Animus' competitors?</summary>
<p>

Any high-level ML/DL libraries, like [Catalyst](https://github.com/catalyst-team/catalyst), [Ignite](https://github.com/pytorch/ignite), [FastAI](https://github.com/fastai/fastai), [Keras](https://github.com/keras-team/keras), etc.

</p>
</details>


<details>
<summary>Why do we need Animus if we have high-level alternatives?</summary>
<p>

Although I find high-level DL frameworks an essential step for the community and the spread of Deep Learning (I have written one by myself), they have a few weaknesses. 

First of all, usually, they are heavily bounded to a single "low-level" DL framework ([Jax](https://github.com/google/jax), [PyTorch](https://github.com/pytorch/pytorch), [Tensorflow](https://github.com/tensorflow/tensorflow)). While ["low-level" frameworks become close each year](https://twitter.com/fchollet/status/1052228463300493312?s=20), high-level frameworks introduce different synthetic sugar, which makes it impossible for a fair comparison, or complementary use, of "low-level" frameworks.


Secondly, high-level frameworks introduce high-level abstractions, which:
- are built with some assumptions in mind, which could be wrong in your case,
- can cause additional bugs - even "low-level" frameworks have quite a lot of them,
- are really hard to debug/extend because of "user-friendly" interfaces and extra integrations.

While these steps could seem unimportant in common cases, like supervised learning with `(features, targets)`, they became more and more important during research and heavy pipeline customization (e.g. privacy-aware multi-node distributed training with custom backpropagation).


Thirdly, many high-level frameworks try to divide ML pipeline into data, hardware, model, etc layers, making it easier for practitioners to start ML experiments and giving teams a tool to separate ML pipeline responsibility between different members. However, while it speeds up the creation of ML pipelines, it disregards that ML experiment results are heavily conditioned on the used model hyperparameters, **and data preprocessing/transformations/sampling**, **and hardware setup**.<br/>
*I found this the main reason why ML experiments fail - you have to focus on the whole data transformation pipeline simultaneously, from raw data through the training process to distributed inference, which is quite hard.*

</p>
</details>


<details>
<summary>What is Animus' purpose?</summary>
<p>

Highlight general "breakpoints" in the ML experiments and give a unified interface for them.

</p>
</details>


<details>
<summary>What is Animus' main application?</summary>
<p>

Research experiments, where you have to define everything on your own to get the results right.

</p>
</details>


<details>
<summary>Does Animus have any requirements?</summary>
<p>

No. That's the case - only pure Python libraries.
PyTorch and Keras could be used for extensions.

</p>
</details>


<details>
<summary>Do you have plans for documentation?</summary>
<p>

No. Animus core is about 300 lines of code, so it's much easier to just read them all, rather than 3000 lines of documentation.

</p>
</details>


#### Demo

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scitator/animus/blob/main/examples/notebooks/colab_ci_cd.ipynb) [Jax/Keras/Sklearn/Torch pipelines](./examples/notebooks/colab_ci_cd.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scitator/animus/blob/main/examples/notebooks/XLA_jax.ipynb) [Jax XLA example](./examples/notebooks/XLA_jax.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scitator/animus/blob/main/examples/notebooks/XLA_torch.ipynb) [Torch XLA example](./examples/notebooks/XLA_torch.ipynb)
