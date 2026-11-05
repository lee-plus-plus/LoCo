import numpy as np
import torch
import torchvision
from os.path import join, normpath
from ..utils import file_cache, ignore_prints, ignore_warnings
from .base import VisualData, VisualDataset

category_name = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    4: "aeroplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "sofa",
    58: "pottedplant",
    59: "bed",
    60: "diningtable",
    61: "toilet",
    62: "tvmonitor",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def load_coco_data(root_dir, year, divide, *, cache=True) -> VisualData:
    assert year in {"2014", "2017"}
    assert divide in {"train", "val"}

    @ignore_warnings()
    @file_cache(cache_dir=".cache/dataset/coco/", enabled=cache)
    def _load_coco_data(year, divide):
        with ignore_prints():
            coco = torchvision.datasets.CocoDetection(
                root=join(root_dir, f"{divide}{year}/"),
                annFile=join(root_dir, f"annotations/instances_{divide}{year}.json"),
            )

        # creating multi-label annotations
        categories = [coco._load_target(coco.ids[i]) for i in range(len(coco))]
        categories = [sorted(list({str(seg_box["category_id"]) for seg_box in elem})) for elem in categories]

        # map category to class-index
        category_map = dict([
            (str(j), i)
            for i, j in enumerate([i for i in range(1, 91) if i not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]])
        ])
        targets = [[category_map[category] for category in elem] for elem in categories]

        # generate label matrix
        num_samples, num_classes = len(targets), len(category_map)
        labels = torch.zeros((num_samples, num_classes)).long()
        for row in range(num_samples):
            labels[row, targets[row]] = 1
        labels = labels.to_sparse_csr()

        # generate image filename (relative path)
        image_filenames = np.array(
            [join(f"{divide}{year}/", elem["file_name"]) for elem in coco.coco.loadImgs(coco.ids)]
        )

        return {"image_filenames": image_filenames, "labels": labels}

    data = _load_coco_data(year, divide)
    image_filenames, labels = data["image_filenames"], data["labels"]
    labels = labels.to_dense()

    # from relative path to abstract path
    image_filenames = [normpath(join(root_dir, filename)) for filename in image_filenames]

    return VisualData(image_filenames, labels, category_name)


class Coco2014Dataset(VisualDataset):
    TYPE = "multi-label"
    SUPPORTED_DIVIDES = ("train", "val")
    CATEGORY_NAME = category_name

    def __init__(self, root_dir, divide):
        data = load_coco_data(root_dir, "2014", divide)
        super().__init__(image_filenames=data.image_filenames, labels=data.labels, category_name=data.category_name)

    def __repr__(self):
        return f"Coco2014Dataset(num_samples={self.num_samples}, num_classes={self.num_classes})"


class Coco2017Dataset(VisualDataset):
    TYPE = "multi-label"
    SUPPORTED_DIVIDES = ("train", "val")
    CATEGORY_NAME = category_name

    def __init__(self, root_dir, divide):
        data = load_coco_data(root_dir, "2017", divide)
        super().__init__(image_filenames=data.image_filenames, labels=data.labels, category_name=data.category_name)

    def __repr__(self):
        return f"Coco2017Dataset(num_samples={self.num_samples}, num_classes={self.num_classes})"
