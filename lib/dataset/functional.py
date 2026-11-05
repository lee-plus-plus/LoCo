import numpy as np
from typing import Callable
from ..utils import local_random, subsample, concat
from .base import VisualDataset, supported_divides


def subsampled(dataset: VisualDataset, begin: float, end: float, shuffle_seed: int | None = None) -> VisualDataset:
    """Returns a subset of the VisualDataset based on the specified range [begin, end],
    with reproducible shuffle if shuffle_seed is not None."""
    assert isinstance(dataset, VisualDataset)
    assert 0.0 <= begin <= end <= 1.0

    indices = np.arange(len(dataset))
    with local_random(shuffle_seed):
        np.random.shuffle(indices)  # reproducible shuffle

    begin = int(len(dataset) * begin)
    end = int(len(dataset) * end)
    indices = indices[begin:end]

    dataset.images = subsample(dataset.images, indices)
    dataset.image_filenames = subsample(dataset.image_filenames, indices)
    dataset.labels = subsample(dataset.labels, indices)

    return dataset


def merged(*datasets: list[VisualDataset]) -> VisualDataset:
    """Merges multiple VisualDataset objects into a single VisualDataset"""
    assert len(datasets) >= 1
    assert len(np.unique([d.images is None for d in datasets])) == 1
    assert len(np.unique([d.image_filenames is None for d in datasets])) == 1

    dataset = datasets[0]
    if len(datasets) == 1:
        return dataset

    dataset.images = concat([d.images for d in datasets])
    dataset.image_filenames = concat([d.image_filenames for d in datasets])
    dataset.labels = concat([d.labels for d in datasets])

    return dataset


def convert_divide(dataset_class: type, divide: str) -> Callable:
    """Enables a dataset_class to fully support divides of {train, val, trainval, test},
    even if the original dataset_class does not explicitly define all these divides.
    This function ensures compatibility by subsampling or merging existing divides
    to create the requested divide."""

    assert divide in supported_divides()
    assert issubclass(dataset_class, VisualDataset)

    dataset_supported_divides: set[str] = set(dataset_class.SUPPORTED_DIVIDES)

    def _subsampled(divide: str, begin: float, end: float) -> Callable:
        """Subsample a dataset from given indices, shuffled"""
        def func(*args, **kwargs):
            dataset = dataset_class(divide=divide, *args, **kwargs)
            dataset = subsampled(dataset, begin=begin, end=end, shuffle_seed=123)
            return dataset

        return func

    def _merged_subsampled(divides: list[str], begin: float, end: float) -> Callable:
        """Merge multiple divide's subsets of a dataset, then subsample it from given indices, shuffled"""
        def func(*args, **kwargs):
            datasets = [dataset_class(divide=d, *args, **kwargs) for d in divides]
            dataset = merged(*datasets)
            dataset = subsampled(dataset, begin=begin, end=end, shuffle_seed=123)
            return dataset

        return func

    if divide == "undivided":
        if "undivided" in dataset_supported_divides:
            complete_divides = ["undivided"]
        elif {"train", "val", "trainval"}.issubset(set(dataset_supported_divides)):
            complete_divides = [e for e in dataset_supported_divides if e != "trainval"]
        else:
            complete_divides = dataset_supported_divides
        return _merged_subsampled(complete_divides, 0.0, 1.0)

    if len(dataset_supported_divides) == 1:
        assert dataset_supported_divides == {"undivided"}

        return {
            "train": _subsampled("undivided", 0.0, 0.6),
            "val": _subsampled("undivided", 0.6, 0.8),
            "trainval": _subsampled("undivided", 0.0, 0.8),
            "test": _subsampled("undivided", 0.8, 1.0),
        }[divide]

    dataset_supported_divides = dataset_supported_divides - {"undivided"}

    if "test" not in dataset_supported_divides:
        assert dataset_supported_divides == {"train", "val"}

        return {
            "train": _subsampled("train", 0.0, 0.75),
            "val": _subsampled("train", 0.75, 1.0),
            "trainval": _subsampled("train", 0.0, 1.0),
            "test": _subsampled("val", 0.0, 1.0),
        }[divide]

    elif "val" not in dataset_supported_divides:
        assert dataset_supported_divides in ({"trainval", "test"}, {"train", "test"})

        TRAINVAL = "trainval" if "trainval" in dataset_supported_divides else "train"
        return {
            "train": _subsampled(TRAINVAL, 0.0, 0.75),
            "val": _subsampled(TRAINVAL, 0.75, 1.0),
            "trainval": _subsampled(TRAINVAL, 0.0, 1.0),
            "test": _subsampled("test", 0.0, 1.0),
        }[divide]

    else:
        assert dataset_supported_divides in ({"train", "val", "test"}, {"train", "val", "trainval", "test"})

        if divide not in dataset_supported_divides:
            return _merged_subsampled(["train", "val"], 0.0, 1.0)
        else:
            return _subsampled(divide, 0.0, 1.0)