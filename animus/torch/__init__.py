# Date: 12/12/2021
# Author: Sergey Kolesnikov (scitator@gmail.com)
# Licence: Apache 2.0
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    IS_TORCH_XLA_AVAILABLE = True
except ImportError:
    IS_TORCH_XLA_AVAILABLE = False

try:
    from accelerate import Accelerator

    IS_ACCELERATE_AVAILABLE = True
except ImportError:
    IS_ACCELERATE_AVAILABLE = False
