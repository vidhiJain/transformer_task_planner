from functools import partial
from typing import Any

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, SequentialLR


class ModuleWrapper:
    def __init__(self, method_name, **kwargs) -> None:
        callable_method = globals()[method_name]
        self._partial = partial(callable_method, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        module = self._partial(*args, **kwds)
        return module
