# Date: 12/12/2021
# Author: Sergey Kolesnikov (scitator@gmail.com)
# Licence: Apache 2.0
from typing import Any

import tensorflow as tf

from animus.callbacks import ICheckpointerCallback
from animus.core import IExperiment


class KerasCheckpointerCallback(ICheckpointerCallback):
    def save(self, exp: IExperiment, obj: Any, logprefix: str) -> str:
        tf.keras.models.save_model(
            obj,
            logprefix,
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None,
        )
        return logprefix


__all__ = [KerasCheckpointerCallback]
