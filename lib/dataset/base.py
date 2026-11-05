import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

# Visual Multi-label Learning Data (For Storage)
@dataclass
class VisualData:
    __slots__ = ["image_filenames", "labels", "category_name"]
    image_filenames: list[str]
    labels: Tensor
    category_name: dict[int, str]

    @property
    def num_samples(self) -> int:
        return self.labels.shape[0]

    @property
    def num_classes(self) -> int:
        return self.labels.shape[1]

    def __post_init__(self):
        assert isinstance(self.image_filenames, list)
        assert isinstance(self.labels, Tensor)
        assert isinstance(self.category_name, dict)
        assert len(self.image_filenames) == self.labels.shape[0]
        assert len(self.category_name) == self.labels.shape[1]
        assert self.labels.dtype == torch.int64


class VisualDataset(Dataset):
    """Visual Multi-label Learning Dataset (For Training)"""

    def __init__(self, labels, category_name, images=None, image_filenames=None):
        super().__init__()
        self.images: list[Image.Image] | Tensor | None = images
        self.image_filenames: list[str] | ndarray | None = image_filenames
        self.labels: Tensor = labels
        self.category_name: dict[int, str] = category_name
        self.getter: ItemGetter = ImageLabelGetter()

    def __getitem__(self, index):
        # dispatch to a normal function, allowing dynamic overriding (add index, add transform, etc)
        # https://stackoverflow.com/questions/57413453/is-it-possible-to-override-getitem-at-instance-level-in-python
        return self.getter(self, index)

    def image(self, index):
        if self.images is not None:
            return self.images[index]
        else:
            return Image.open(self.image_filenames[index]).convert("RGB")

    def label(self, index):
        return self.labels[index]

    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={self.num_samples}, num_classes={self.num_classes})"

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return self.labels.shape[0]

    @property
    def num_classes(self):
        return self.labels.shape[1]

    def __post_init__(self):
        assert isinstance(self.images, (list, Tensor, None))
        assert isinstance(self.image_filenames, (list, ndarray, None))
        assert isinstance(self.labels, Tensor)
        assert isinstance(self.category_name, dict)
        assert self.images is not None or self.image_filenames is not None
        assert self.labels.dtype == torch.int64


class ItemGetter:
    """Item Getter for VisualDataset.

    By inheriting it, VisualDataset can customize its batch, e.g., transform and index.
    It is expected to be a stateless callable class.

    Usage:
    >>> class IndexedGetter(ItermGetter):
    >>>     def forward(self, dataset, index):
    >>>         return dataset.image(index), dataset.label(index)
    >>> dataset.getter = IndexedGetter()
    """

    def __call__(self, dataset, index, *args, **kwargs):
        return self.forward(dataset, index)

    def forward(self, dataset, index):
        raise NotImplemented()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ImageLabelGetter:
    def forward(self, dataset, index):
        return dataset.image(index), dataset.label(index)


SUPPORTED_DIVIDES = ("train", "val", "trainval", "test", "undivided")


def supported_divides():
    return SUPPORTED_DIVIDES
