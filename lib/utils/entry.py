import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import os
import io
import sys
import pynvml
import importlib
import inspect
from omegaconf import OmegaConf, DictConfig


def init_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_cuda_environment():
    cudnn.benchmark = True
    cudnn.deterministic = True


def reload(obj, override=True):
    """
    Reload the module associated with the given object and return the updated object.

    Args:
        obj: The object to reload (can be a module, class, function, instance method, etc.).
        override (bool): Whether to overwrite the attributes of the original object (e.g., `__code__` for functions).

    Returns:
        The updated object.

    Raises:
        ValueError: If the object is not associated with a module or lacks a `__name__` attribute.
    """
    # Get the module associated with the object
    module = inspect.getmodule(obj)
    if module is None:
        raise ValueError(f"Cannot reload {obj}: it is not associated with a module.")

    module_name = module.__name__

    # Reload the module
    module = importlib.import_module(module_name)
    importlib.reload(module)

    # If the object is a module itself, return the reloaded module
    if inspect.ismodule(obj):
        return module

    # Get the name of the object
    obj_name = obj.__name__ if hasattr(obj, "__name__") else None
    if obj_name is None:
        raise ValueError(f"Cannot reload {obj}: it does not have a __name__ attribute.")

    # Get the updated object from the reloaded module
    updated_obj = getattr(module, obj_name)

    if override:
        if inspect.isfunction(obj):
            obj.__code__ = updated_obj.__code__

        elif inspect.isclass(obj):
            for attr_name, attr_value in updated_obj.__dict__.items():
                if not attr_name.startswith("__"):  # Avoid overwriting special methods
                    setattr(obj, attr_name, attr_value)

        elif inspect.ismethod(obj):
            # Get the class of the instance method
            class_obj = obj.__self__.__class__
            # Update the class reference
            for attr_name, attr_value in updated_obj.__dict__.items():
                if not attr_name.startswith("__"):  # Avoid overwriting special methods
                    setattr(class_obj, attr_name, attr_value)

    return updated_obj


def best_gpu():
    pynvml.nvmlInit()
    mem_free = [
        pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free
        for i in range(pynvml.nvmlDeviceGetCount())
    ]
    return mem_free.index(max(mem_free))


def is_redirected():
    """Check if the standard output is redirected."""
    try:
        return not os.isatty(sys.stdout.fileno())
    except io.UnsupportedOperation as e:
        return False


def parse_from_cli() -> DictConfig:
    return OmegaConf.to_container(OmegaConf.from_cli(), resolve=True)
