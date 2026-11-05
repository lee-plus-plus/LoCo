import numpy as np
import torch
import h5py
import json
from os.path import join, normpath
from ..utils import file_cache, ignore_warnings
from .base import VisualData, VisualDataset


category_name = {
    0: "airplane",
    1: "animal",
    2: "arm",
    3: "bag",
    4: "banana",
    5: "basket",
    6: "beach",
    7: "bear",
    8: "bed",
    9: "bench",
    10: "bike",
    11: "bird",
    12: "board",
    13: "boat",
    14: "book",
    15: "boot",
    16: "bottle",
    17: "bowl",
    18: "box",
    19: "boy",
    20: "branch",
    21: "building",
    22: "bus",
    23: "cabinet",
    24: "cap",
    25: "car",
    26: "cat",
    27: "chair",
    28: "child",
    29: "clock",
    30: "coat",
    31: "counter",
    32: "cow",
    33: "cup",
    34: "curtain",
    35: "desk",
    36: "dog",
    37: "door",
    38: "drawer",
    39: "ear",
    40: "elephant",
    41: "engine",
    42: "eye",
    43: "face",
    44: "fence",
    45: "finger",
    46: "flag",
    47: "flower",
    48: "food",
    49: "fork",
    50: "fruit",
    51: "giraffe",
    52: "girl",
    53: "glass",
    54: "glove",
    55: "guy",
    56: "hair",
    57: "hand",
    58: "handle",
    59: "hat",
    60: "head",
    61: "helmet",
    62: "hill",
    63: "horse",
    64: "house",
    65: "jacket",
    66: "jean",
    67: "kid",
    68: "kite",
    69: "lady",
    70: "lamp",
    71: "laptop",
    72: "leaf",
    73: "leg",
    74: "letter",
    75: "light",
    76: "logo",
    77: "man",
    78: "men",
    79: "motorcycle",
    80: "mountain",
    81: "mouth",
    82: "neck",
    83: "nose",
    84: "number",
    85: "orange",
    86: "pant",
    87: "paper",
    88: "paw",
    89: "people",
    90: "person",
    91: "phone",
    92: "pillow",
    93: "pizza",
    94: "plane",
    95: "plant",
    96: "plate",
    97: "player",
    98: "pole",
    99: "post",
    100: "pot",
    101: "racket",
    102: "railing",
    103: "rock",
    104: "roof",
    105: "room",
    106: "screen",
    107: "seat",
    108: "sheep",
    109: "shelf",
    110: "shirt",
    111: "shoe",
    112: "short",
    113: "sidewalk",
    114: "sign",
    115: "sink",
    116: "skateboard",
    117: "ski",
    118: "skier",
    119: "sneaker",
    120: "snow",
    121: "sock",
    122: "stand",
    123: "street",
    124: "surfboard",
    125: "table",
    126: "tail",
    127: "tie",
    128: "tile",
    129: "tire",
    130: "toilet",
    131: "towel",
    132: "tower",
    133: "track",
    134: "train",
    135: "tree",
    136: "truck",
    137: "trunk",
    138: "umbrella",
    139: "vase",
    140: "vegetable",
    141: "vehicle",
    142: "wave",
    143: "wheel",
    144: "window",
    145: "windshield",
    146: "wing",
    147: "wire",
    148: "woman",
    149: "zebra",
}


def load_vg150_data(root_dir, divide, *, cache=True) -> VisualData:
    assert divide in {"train", "val"}

    @ignore_warnings()
    @file_cache(cache_dir=".cache/dataset/vg150/", enabled=cache)
    def _load_vg150_data(divide):
        info = json.load(open(join(root_dir, "meta/VG-SGG-dicts-with-attri.json"), "r"))
        img_data = json.load(open(join(root_dir, "meta/image_data.json"), "r"))
        attri_data = h5py.File(join(root_dir, "meta/VG-SGG-with-attri.h5"), "r")

        box_labels = attri_data["labels"]
        img_to_first_box = attri_data["img_to_first_box"]
        img_to_last_box = attri_data["img_to_last_box"]
        _category_name = {int(key) - 1: value for key, value in info["idx_to_label"].items()}
        image_splits = list([elem.item() for elem in attri_data['split']])  # 0: train, 1: test, 2: val (only 0 and 2)

        image_filenames = [f"images/{elem['image_id']}.jpg" for elem in img_data]
        targets = [sorted(list(set([(int(elem[0]) - 1) for elem in box_labels[st : ed + 1]]))) for st, ed in zip(img_to_first_box, img_to_last_box)]

        if divide == "train":
            indices = [i for i, elem in enumerate(image_splits) if elem == 0]
        elif divide == "val":
            indices = [i for i, elem in enumerate(image_splits) if elem == 2]
        else:
            indices = [i for i, elem in enumerate(image_splits)]
        image_filenames, targets = [image_filenames[i] for i in indices], [targets[i] for i in indices]

        # generate label matrix
        num_samples, num_classes = len(targets), len(_category_name)
        labels = torch.zeros((num_samples, num_classes)).long()
        for row in range(num_samples):
            labels[row, targets[row]] = 1
        labels = labels.to_sparse_csr()

        # generate image filename (relative path)
        image_filenames = np.array(image_filenames)

        return {"image_filenames": image_filenames, "labels": labels}

    data = _load_vg150_data(divide)
    image_filenames, labels = data["image_filenames"], data["labels"]
    labels = labels.to_dense()

    # from relative path to abstract path
    image_filenames = [normpath(join(root_dir, filename)) for filename in image_filenames]

    return VisualData(image_filenames, labels, category_name)


class Vg150Dataset(VisualDataset):
    TYPE = "multi-label"
    SUPPORTED_DIVIDES = ("train", "val")
    CATEGORY_NAME = category_name

    def __init__(self, root_dir, divide):
        data = load_vg150_data(root_dir, divide)
        super().__init__(image_filenames=data.image_filenames, labels=data.labels, category_name=data.category_name)

    def __repr__(self):
        return f"Vg150Dataset(num_samples={self.num_samples}, num_classes={self.num_classes})"
