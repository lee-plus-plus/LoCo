import numpy as np
import torch
import torchvision
from os.path import join, normpath, relpath
from ..utils import file_cache, ignore_prints
from .base import VisualData, VisualDataset

category_name = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor",
}


def load_voc_data(root_dir, year, divide, *, cache=True) -> VisualData:
    assert year in ("2007", "2012")
    assert divide in ("train", "val", "trainval", "test")
    assert (year, divide) != ("2012", "test")

    @file_cache(cache_dir=".cache/dataset/voc/", enabled=cache)
    def _load_voc_data(year, divide):
        with ignore_prints():
            voc = torchvision.datasets.VOCDetection(root=root_dir, year=year, image_set=divide, download=False)

        # creating multi-label annotations
        annotations = [tgt["annotation"]["object"] for img, tgt in voc]
        categories = [sorted(list({str(seg_box["name"]) for seg_box in elem})) for elem in annotations]
        category_map = dict(zip(category_name.values(), category_name.keys()))
        targets = [[category_map[category] for category in elem] for elem in categories]

        # generate label matrix
        num_samples, num_classes = len(voc), len(category_map)
        labels = torch.zeros((num_samples, num_classes)).long()
        for row in range(num_samples):
            labels[row, targets[row]] = 1

        # generate image filename (relative path)
        image_filenames = np.array([relpath(filename, root_dir) for filename in voc.images])

        return {"image_filenames": image_filenames, "labels": labels}

    data = _load_voc_data(year, divide)
    image_filenames, labels = data["image_filenames"], data["labels"]
    
    # from relative path to abstract path
    image_filenames = [normpath(join(root_dir, filename)) for filename in image_filenames]

    return VisualData(image_filenames, labels, category_name)


class Voc2007Dataset(VisualDataset):
    TYPE = "multi-label"
    SUPPORTED_DIVIDES = ("train", "val", "trainval", "test")
    CATEGORY_NAME = category_name

    def __init__(self, root_dir, divide):
        data = load_voc_data(root_dir, "2007", divide)
        super().__init__(
            image_filenames=data.image_filenames,
            labels=data.labels,
            category_name=data.category_name,
        )


class Voc2012Dataset(VisualDataset):
    TYPE = "multi-label"
    SUPPORTED_DIVIDES = ("train", "val", "trainval")
    CATEGORY_NAME = category_name

    def __init__(self, root_dir, divide):
        data = load_voc_data(root_dir, "2012", divide)
        super().__init__(
            image_filenames=data.image_filenames,
            labels=data.labels,
            category_name=data.category_name,
        )
