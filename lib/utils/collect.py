import torch
from torch import Tensor
from torchmetrics import Metric


def to_cpu(batch):
    if callable(batch) and not isinstance(batch, Tensor):
        def wrapper(*args, **kwargs):
            return to_cpu(batch(*args, **kwargs))
        return wrapper
    if isinstance(batch, Tensor):
        return batch.cpu()
    elif isinstance(batch, (tuple, list)):
        return type(batch)([elem.cpu() for elem in batch])
    elif isinstance(batch, dict):
        return {key: value.cpu() for key, value in batch.items()}


class Collector(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("indices", default=[], dist_reduce_fx=None)
        # No fixed states; added dynamically in update

    def update(self, indices=None, **kwargs):
        if indices is not None:
            self.indices.append(indices.detach().to(self.device))

        for key, value in kwargs.items():
            if not hasattr(self, key):
                self.add_state(key, default=[], dist_reduce_fx=None)
            getattr(self, key).append(value.detach().to(self.device))

    def compute(self):
        # Use 'indices' (if present) to determine sorting order
        if hasattr(self, 'indices') and len(self.indices) > 0:
            idxs = torch.cat(self.indices, dim=0)
            sorted_indices = torch.argsort(idxs)
        else:
            sorted_indices = None

        result = {}
        # Iterate over instance attributes and select those stored as lists (states)
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if key == "indices":
                continue
            if isinstance(value, list) and len(value) > 0:
                cat_value = torch.cat(value, dim=0)
                result[key] = cat_value[sorted_indices] if sorted_indices is not None else cat_value

        return result
