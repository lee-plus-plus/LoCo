import numpy as np
import torch
import random
from typing import TypeVar
import numpy as np
import torch
from typing import Any
from .functional import local_random

T = TypeVar("T", np.ndarray, torch.Tensor, list)


def to_tensor(X: T, copy: bool = False) -> torch.Tensor:
    """transform np.ndarry / torch.Tensor to torch.Tensor"""
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X)
    if isinstance(X, torch.Tensor):
        return X.clone() if copy else X
    else:
        raise ValueError(f"cannot convert {type(X)} to torch.Tensor")


def shuffled(data: T) -> T:
    if isinstance(data, np.ndarray):
        idx = np.random.permutation(len(data))
        return data[idx]
    elif isinstance(data, torch.Tensor):
        idx = torch.randperm(len(data))
        return data[idx]
    elif isinstance(data, list):
        data_copy = data[:]
        random.shuffle(data_copy)
        return data_copy
    else:
        raise TypeError(f"Unsupported data type {type(data)}")


def one_hot(x: T, num_classes: int) -> T:
    if isinstance(x, np.ndarray):
        return np.eye(num_classes)[x]
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.one_hot(x, num_classes)
    if isinstance(x, list):
        return [[1 if i == label else 0 for i in range(num_classes)] for label in x]
    else:
        raise ValueError(f"Unsupported data type {type(x)}")


def get_split_indices(num_samples: int, split_ratio: list[float], shuffle_seed: int | None = None) -> list[list[int]]:
    """Splits a dataset of indices into multiple subsets according to the specified split ratios.

    >>> train_idx, test_idx = get_split_indices(100, [0.8, 0.2])
    >>> len(train_idx), len(test_idx)
    (80, 20)
    """
    indices = list(range(num_samples))
    split_ratio = (np.cumsum([0] + split_ratio) * num_samples).astype(int)

    with local_random(shuffle_seed):
        np.random.shuffle(indices)  # reproducible shuffle

    splited_indices = [indices[st:ed] for st, ed in zip(split_ratio[:-1], split_ratio[1:])]
    return splited_indices


def subsample(elem_list: Any, idxs: np.ndarray):
    """Subsamples elements from a list or array based on given indices"""
    if elem_list is None:
        return None
    elif isinstance(elem_list, (torch.Tensor, np.ndarray)):
        return elem_list[idxs]
    elif isinstance(elem_list, list):
        return [elem_list[i] for i in idxs]
    else:
        return type(elem_list)([elem_list[i] for i in idxs])


def concat(elem_list: list[Any]) -> Any:
    """Concatenates elements of a specific type (Tensor, ndarray, or list)"""
    pivot = elem_list[0]
    if pivot is None:
        return None
    elif isinstance(pivot, torch.Tensor):
        return torch.cat(elem_list, dim=0)
    elif isinstance(pivot, np.ndarray):
        return np.concatenate(elem_list, axis=0)
    elif isinstance(pivot, list):
        return sum(elem_list, start=[])
    else:
        return type(pivot)(sum([list(elem) for elem in elem_list], start=[]))
