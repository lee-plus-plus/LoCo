from os.path import join, expanduser

from .voc import Voc2007Dataset
from .coco import Coco2014Dataset
from .nuswide import NusWideDataset
from .vg150 import Vg150Dataset

from .customize import customize

from .base import VisualDataset, supported_divides
from .functional import convert_divide

SUPPORTED_DATASETS = {
    "voc2007": Voc2007Dataset,
    "coco2014": Coco2014Dataset,
    "nuswide": NusWideDataset,
    "vg150": Vg150Dataset,
}


def supported_multilabel_datasets():
    return [k for k, v in SUPPORTED_DATASETS.items() if v.TYPE == "multi-label"]


def supported_multiclass_datasets():
    return [k for k, v in SUPPORTED_DATASETS.items() if v.TYPE == "multi-class"]


def supported_datasets():
    return supported_multilabel_datasets() + supported_multiclass_datasets()


def create_dataset(
    name: str,
    divide: str,
    base: str = expanduser("~/dataset"),
    **kwargs,
) -> VisualDataset:
    assert name in supported_datasets()
    assert divide in supported_divides()

    root_dir = join(base, name)  # e.g., ~/dataset/voc2007/ for name="voc2007"
    dataset_class = SUPPORTED_DATASETS[name]
    dataset_class = convert_divide(dataset_class, divide)  # to support any divide
    dataset = dataset_class(root_dir=root_dir)

    dataset = customize(dataset, **kwargs)  # to add fancy functions

    return dataset
