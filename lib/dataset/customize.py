import builtins
from random import sample, choices
from typing import Literal
from math import ceil
import torch
from ..utils import get_split_indices, shuffled, subsample, local_random
from .base import VisualDataset, ItemGetter
from .noise import add_partial_noise, add_imbalanced_noise


class IndexGetter(ItemGetter):
    def __init__(self, getter):
        super().__init__()
        self.getter = getter

    def forward(self, dataset, index):
        return self.getter(dataset, index) + (index,)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.getter})"


class PartialLabelGetter(ItemGetter):
    def __init__(self, getter):
        super().__init__()
        self.getter = getter

    def forward(self, dataset, index):
        return self.getter(dataset, index) + (dataset.partial_labels[index],)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.getter})"


class NoisyLabelGetter(ItemGetter):
    def __init__(self, getter):
        super().__init__()
        self.getter = getter

    def forward(self, dataset, index):
        return self.getter(dataset, index) + (dataset.partial_labels[index],)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.getter})"


class TransformedImageLabelGetter(ItemGetter):
    def forward(self, dataset, index):
        image = dataset.image(index)
        image = dataset.transform(image) if dataset.transform is not None else image
        label = dataset.label(index)
        return image, label


def downsampled(dataset: VisualDataset, downsample_ratio: float) -> VisualDataset:
    """Downsample the dataset according to the given ratio"""

    assert 0.0 <= downsample_ratio <= 1.0
    indices = get_split_indices(dataset.num_samples, [downsample_ratio], shuffle_seed=123)[0]

    num_samples = dataset.num_samples
    for attr in dir(dataset):
        value = getattr(dataset, attr)
        if (
            (not attr.startswith("_"))
            and (not attr.isupper())
            and hasattr(value, "__len__")
            and len(value) == num_samples
            and not isinstance(getattr(type(dataset), attr, None), property)
        ):
            setattr(dataset, attr, subsample(value, indices))

    return dataset


def upsampled(dataset: VisualDataset, upsample_ratio: float) -> VisualDataset:
    """Upsample the dataset according to the given ratio"""

    assert upsample_ratio >= 1.0
    total_samples = int(dataset.num_samples * upsample_ratio)
    indices = torch.arange(dataset.num_samples).repeat(ceil(upsample_ratio))[:total_samples]

    num_samples = dataset.num_samples
    for attr in dir(dataset):
        value = getattr(dataset, attr)
        if (
            (not attr.startswith("_"))
            and (not attr.isupper())
            and not isinstance(value, dict)
            and hasattr(value, "__len__")
            and len(value) == num_samples
            and not isinstance(getattr(type(dataset), attr, None), property)
        ):
            setattr(dataset, attr, subsample(value, indices))

    return dataset


def indexed(dataset: VisualDataset) -> VisualDataset:
    """let the dataset iterate with index"""

    dataset.getter = IndexGetter(dataset.getter)
    return dataset


def partially_noisy(dataset: VisualDataset, noise_rate: float) -> VisualDataset:
    """let the dataset iterate with partial labels"""

    dataset.partial_labels = add_partial_noise(dataset.labels, noise_rate)
    dataset.getter = PartialLabelGetter(dataset.getter)
    return dataset


def imbalancedly_noisy(dataset: VisualDataset, pos_noise_rate: float, neg_noise_rate: float) -> VisualDataset:
    """let the dataset iterate with multi-label noisy labels"""

    dataset.noisy_labels = add_imbalanced_noise(dataset.labels, pos_noise_rate, neg_noise_rate)
    dataset.getter = NoisyLabelGetter(dataset.getter)
    return dataset


def transformed(dataset: VisualDataset, transform: object) -> VisualDataset:
    """let the dataset iterate with transformed images"""

    dataset.transform = transform
    dataset.getter = TransformedImageLabelGetter()
    return dataset


def few_shoted(
    dataset: VisualDataset, num_shots: int, strategy: Literal["any", "onehot-only", "onehot-prior"] = "any", 
    shuffle_seed: int | None = None, repeat: bool = False
) -> VisualDataset:
    """downsample the multi-label dataset into a few-shot dataset"""

    num_classes = dataset.num_classes

    def _clamp(value, min=None, max=None):
        if min is not None:
            value = builtins.max(value, min)
        if max is not None:
            value = builtins.min(value, max)
        return value

    def _sample(population, k):
        if k <= len(population):
            # No repeat, just sample
            return sample(population, k)
        else:
            # Sample all unique first, then repeat sample for the rest
            result = sample(population, len(population))
            if repeat:
                result += choices(population, k=k - len(population))
                assert len(result) == k
            return result

    indices = []
    with local_random(seed=shuffle_seed):
        for cls in range(num_classes):
            if strategy == "any" or dataset.TYPE == "multi-class":
                candidates = (dataset.labels[:, cls] == 1).nonzero().flatten().tolist()
                choosen_indices = _sample(candidates, k=_clamp(num_shots, min=0))

            elif strategy == "onehot-only":
                candidates = ((dataset.labels.sum(dim=1) == 1) & (dataset.labels[:, cls] == 1)).nonzero().flatten().tolist()
                choosen_indices = _sample(candidates, k=_clamp(num_shots, min=0))

            elif strategy == "onehot-prior":
                candidates1 = ((dataset.labels[:, cls] == 1) & (dataset.labels.sum(dim=1) == 1)).nonzero().flatten().tolist()
                candidates2 = ((dataset.labels[:, cls] == 1) & (dataset.labels.sum(dim=1) != 1)).nonzero().flatten().tolist()
                choosen_indices1 = _sample(candidates1, k=_clamp(num_shots, min=0, max=len(candidates1)))
                choosen_indices2 = _sample(candidates2, k=_clamp(num_shots - len(choosen_indices1), min=0))
                choosen_indices = choosen_indices1 + choosen_indices2

            indices.append(choosen_indices)
        indices = torch.tensor(sum(indices, start=[])).long()
        if not repeat:
            indices = indices.unique()

    for attr in dir(dataset):
        value = getattr(dataset, attr)
        if (
            (not attr.startswith("_"))
            and (not attr.isupper())
            and hasattr(value, "__len__")
            and len(value) == dataset.num_samples
            and not isinstance(getattr(type(dataset), attr, None), property)
        ):
            setattr(dataset, attr, subsample(value, indices))
    return dataset


def customize(
    dataset: VisualDataset,
    *,
    downsample_ratio: float = 1.0,
    add_index: bool = False,
    add_partial_noise: bool = False,
    add_imbalanced_noise: bool = False,
    noise_rate: float = 0.0,
    pos_noise_rate: float = 0.0,
    neg_noise_rate: float = 0.0,
    transform: object | None = None,
):
    if downsample_ratio != 1.0:
        dataset = downsampled(dataset, downsample_ratio)
    if transform:
        dataset = transformed(dataset, transform)
    if add_partial_noise:
        dataset = partially_noisy(dataset, noise_rate)
    if add_imbalanced_noise:
        dataset = imbalancedly_noisy(dataset, pos_noise_rate, neg_noise_rate)
    if add_index:
        dataset = indexed(dataset)

    return dataset
