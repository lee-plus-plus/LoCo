from copy import deepcopy
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lib.backbone import DenseVisualEncoder, create_clip
from lib.backbone.clip.model import CLIP
from lib import transforms as t
from lib.dataset import create_dataset
from lib.trainer.base import BaseConfig, BaseTrainer, entry
from lib.trainer.scclip.model import ScClipVisionTransformer
from lib.utils.slide_window import SlideWindowCropper as GridCropper


class DenseSCClipEncoder(nn.Module):
    def __init__(
        self,
        clip_model: CLIP,
    ):
        super().__init__()
        self.module = deepcopy(clip_model.visual)
        self.module.__class__ = ScClipVisionTransformer
        self.module.post_init(
            anomaly_resolve=False,
            multi_layer_fuse=True,
            attn_enhance=True,
            feat_aggregate=True,
            cls_token_require=True,
        )

    def forward(self, image: Tensor, normalize: bool = True, spatialize: bool = True) -> Tensor:
        x = self.module(image)
        if normalize:
            x = x / x.norm(dim=-1, keepdim=True)

        global_feats, local_feats = x[:, 0, :], x[:, 1:, :]
        if spatialize:
            batch_size, num_tokens, dim = local_feats.shape
            height = width = int(num_tokens**0.5)
            local_feats = local_feats.view(batch_size, height, width, dim)

        return global_feats, local_feats


def compatibility_matrix(classnames):
    NUS_COMPATIBILITY: dict[str, list[str]] = {
        "airport": ["buildings", "cars", "flags", "plane", "road", "sign", "street", "vehicle"],
        "animal": [
            "bear",
            "birds",
            "cat",
            "coral",
            "cow",
            "dog",
            "elk",
            "fish",
            "fox",
            "horses",
            "tiger",
            "whales",
            "zebra",
        ],
        "beach": ["boats", "ocean", "rocks", "sand", "sky", "sunset", "surf", "swimmers", "water"],
        "bear": ["animal"],
        "birds": ["animal"],
        "boats": ["beach", "harbor", "military", "reflection", "vehicle"],
        "book": ["map"],
        "bridge": ["cityscape", "railroad", "road", "street"],
        "buildings": [
            "airport",
            "castle",
            "cityscape",
            "earthquake",
            "fire",
            "house",
            "reflection",
            "street",
            "temple",
            "tower",
            "town",
            "window",
        ],
        "cars": [
            "airport",
            "cityscape",
            "fire",
            "frost",
            "military",
            "police",
            "reflection",
            "road",
            "street",
            "town",
            "vehicle",
        ],
        "castle": ["buildings", "tower", "window"],
        "cat": ["animal"],
        "cityscape": [
            "bridge",
            "buildings",
            "cars",
            "clouds",
            "nighttime",
            "person",
            "reflection",
            "road",
            "sign",
            "sky",
            "statue",
            "street",
            "sunset",
            "tower",
            "vehicle",
            "window",
        ],
        "clouds": ["cityscape", "moon", "mountain", "nighttime", "rainbow", "reflection", "sky", "sunset"],
        "computer": ["map"],
        "coral": ["animal"],
        "cow": ["animal"],
        "dancing": ["person", "wedding"],
        "dog": ["animal"],
        "earthquake": ["buildings", "fire", "house", "road", "rocks", "street", "town"],
        "elk": ["animal"],
        "fire": ["buildings", "cars", "earthquake", "grass", "house", "plants", "road", "street", "tree", "vehicle"],
        "fish": ["animal"],
        "flags": ["airport", "military", "protest"],
        "flowers": ["frost", "garden", "plants", "wedding"],
        "food": ["wedding"],
        "fox": ["animal"],
        "frost": ["cars", "flowers", "grass", "leaf", "plants", "road", "rocks", "tree", "window"],
        "garden": ["flowers", "grass", "house", "leaf", "plants", "statue", "tree"],
        "glacier": ["mountain", "snow"],
        "grass": ["fire", "frost", "garden", "plants", "snow", "soccer", "sports"],
        "harbor": ["boats", "ocean", "reflection", "sky", "sunset", "water"],
        "horses": ["animal"],
        "house": ["buildings", "earthquake", "fire", "garden", "town", "window"],
        "lake": ["reflection", "sunset", "water"],
        "leaf": ["frost", "garden", "plants"],
        "map": ["book", "computer"],
        "military": ["boats", "cars", "flags", "person", "plane", "train", "vehicle"],
        "moon": ["clouds", "nighttime", "reflection", "sky"],
        "mountain": ["clouds", "glacier", "reflection", "rocks", "sky", "snow", "sunset", "tree", "valley"],
        "nighttime": ["cityscape", "clouds", "moon", "sky"],
        "ocean": ["beach", "harbor", "reflection", "sunset", "surf", "water"],
        "person": [
            "cityscape",
            "dancing",
            "military",
            "police",
            "protest",
            "reflection",
            "running",
            "soccer",
            "sports",
            "street",
            "surf",
            "swimmers",
            "tattoo",
            "town",
            "wedding",
        ],
        "plane": ["airport", "military", "vehicle"],
        "plants": ["fire", "flowers", "frost", "garden", "grass", "leaf", "snow", "tree"],
        "police": ["cars", "person", "protest", "street", "vehicle"],
        "protest": ["flags", "person", "police", "sign", "street"],
        "railroad": ["bridge", "train"],
        "rainbow": ["clouds", "sky", "waterfall"],
        "reflection": [
            "boats",
            "buildings",
            "cars",
            "cityscape",
            "clouds",
            "harbor",
            "lake",
            "moon",
            "mountain",
            "ocean",
            "person",
            "sky",
            "sun",
            "sunset",
            "tree",
            "water",
            "window",
        ],
        "road": [
            "airport",
            "bridge",
            "cars",
            "cityscape",
            "earthquake",
            "fire",
            "frost",
            "running",
            "sign",
            "snow",
            "street",
            "town",
            "vehicle",
        ],
        "rocks": ["beach", "earthquake", "frost", "mountain", "snow", "waterfall"],
        "running": ["person", "road", "sports"],
        "sand": ["beach"],
        "sign": ["airport", "cityscape", "protest", "road", "street", "town"],
        "sky": [
            "beach",
            "cityscape",
            "clouds",
            "harbor",
            "moon",
            "mountain",
            "nighttime",
            "rainbow",
            "reflection",
            "sun",
            "sunset",
            "town",
        ],
        "snow": ["glacier", "grass", "mountain", "plants", "road", "rocks", "tree"],
        "soccer": ["grass", "person", "sports"],
        "sports": ["grass", "person", "running", "soccer", "surf", "swimmers"],
        "statue": ["cityscape", "garden", "temple", "town"],
        "street": [
            "airport",
            "bridge",
            "buildings",
            "cars",
            "cityscape",
            "earthquake",
            "fire",
            "person",
            "police",
            "protest",
            "road",
            "sign",
            "town",
            "vehicle",
            "window",
        ],
        "sun": ["reflection", "sky", "sunset"],
        "sunset": [
            "beach",
            "cityscape",
            "clouds",
            "harbor",
            "lake",
            "mountain",
            "ocean",
            "reflection",
            "sky",
            "sun",
            "water",
        ],
        "surf": ["beach", "ocean", "person", "sports", "water"],
        "swimmers": ["beach", "person", "sports"],
        "tattoo": ["person"],
        "temple": ["buildings", "statue", "tower", "window"],
        "tiger": ["animal"],
        "tower": ["buildings", "castle", "cityscape", "temple", "town", "window"],
        "town": [
            "buildings",
            "cars",
            "earthquake",
            "house",
            "person",
            "road",
            "sign",
            "sky",
            "statue",
            "street",
            "tower",
            "vehicle",
            "window",
        ],
        "toy": [],
        "train": ["military", "railroad", "vehicle"],
        "tree": ["fire", "frost", "garden", "mountain", "plants", "reflection", "snow"],
        "valley": ["mountain"],
        "vehicle": [
            "airport",
            "boats",
            "cars",
            "cityscape",
            "fire",
            "military",
            "plane",
            "police",
            "road",
            "street",
            "town",
            "train",
        ],
        "water": ["beach", "harbor", "lake", "ocean", "reflection", "sunset", "surf", "waterfall"],
        "waterfall": ["rainbow", "rocks", "water"],
        "wedding": ["dancing", "flowers", "food", "person"],
        "whales": ["animal"],
        "window": [
            "buildings",
            "castle",
            "cityscape",
            "frost",
            "house",
            "reflection",
            "street",
            "temple",
            "tower",
            "town",
        ],
        "zebra": ["animal"],
    }

    COCO_COMPATIBILITY: dict[str, list[str]] = {
        "person": [
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "cell phone",
            "scissors",
            "hair drier",
            "toothbrush",
        ],
        "bicycle": [],
        "car": [],
        "motorbike": [],
        "aeroplane": [],
        "bus": [],
        "train": [],
        "truck": [],
        "boat": [],
        "traffic light": [],
        "fire hydrant": [],
        "stop sign": [],
        "parking meter": [],
        "bench": [],
        "bird": [],
        "cat": [],
        "dog": [],
        "horse": [],
        "sheep": [],
        "cow": [],
        "elephant": [],
        "bear": [],
        "zebra": [],
        "giraffe": [],
        "backpack": ["person"],
        "umbrella": ["person"],
        "handbag": ["person"],
        "tie": ["person"],
        "suitcase": ["person"],
        "frisbee": ["person"],
        "skis": ["person"],
        "snowboard": ["person"],
        "sports ball": ["person"],
        "kite": ["person"],
        "baseball bat": ["person"],
        "baseball glove": ["person"],
        "skateboard": ["person"],
        "surfboard": ["person"],
        "tennis racket": ["person"],
        "bottle": ["person"],
        "wine glass": ["person", "cup"],
        "cup": ["person", "wine glass"],
        "fork": ["person"],
        "knife": ["person"],
        "spoon": ["person"],
        "bowl": ["person"],
        "banana": [],
        "apple": [],
        "sandwich": ["hot dog"],
        "orange": [],
        "broccoli": [],
        "carrot": [],
        "hot dog": ["sandwich"],
        "pizza": [],
        "donut": [],
        "cake": [],
        "chair": [],
        "sofa": [],
        "pottedplant": ["vase"],
        "bed": [],
        "diningtable": [],
        "toilet": [],
        "tvmonitor": [],
        "laptop": ["keyboard"],
        "mouse": [],
        "remote": [],
        "keyboard": ["laptop"],
        "cell phone": ["person"],
        "microwave": ["oven"],
        "oven": ["microwave"],
        "toaster": [],
        "sink": [],
        "refrigerator": [],
        "book": [],
        "clock": [],
        "vase": ["pottedplant"],
        "scissors": ["person"],
        "teddy bear": [],
        "hair drier": ["person"],
        "toothbrush": ["person"],
    }

    VOC_COMPATIBILITY: dict[str, list[str]] = {
        "aeroplane": [],
        "bicycle": ["person"],
        "bird": [],
        "boat": ["person"],
        "bottle": ["person", "diningtable"],
        "bus": ["person"],
        "car": ["person"],
        "cat": [],
        "chair": ["person"],
        "cow": [],
        "diningtable": ["bottle"],
        "dog": [],
        "horse": ["person"],
        "motorbike": ["person"],
        "person": [
            "bicycle",
            "boat",
            "bottle",
            "bus",
            "car",
            "chair",
            "horse",
            "motorbike",
            "sofa",
            "train",
            "tvmonitor",
        ],
        "pottedplant": [],
        "sheep": [],
        "sofa": ["person"],
        "train": ["person"],
        "tvmonitor": ["person"],
    }

    VG_COMPATIBILITY: dict[str, list[str]] = {
        "airplane": ["engine", "letter", "logo", "number", "plane", "tire", "vehicle", "wheel", "window", "wing"],
        "animal": ["bear", "bird", "cat", "cow", "dog", "ear", "elephant", "eye", "face", "giraffe", "head", "horse", "leg", "mouth", "neck", "nose", "paw", "sheep", "tail", "trunk", "wing", "zebra"],
        "arm": ["boy", "child", "coat", "finger", "girl", "glove", "guy", "hand", "jacket", "kid", "lady", "man", "men", "people", "person", "player", "shirt", "skier", "woman"],
        "bag": ["boy", "child", "girl", "guy", "handle", "kid", "lady", "letter", "logo", "man", "men", "number", "people", "person", "player", "skier", "woman"],
        "banana": ["food", "fruit"],
        "basket": ["handle"],
        "beach": ["boat", "people", "person", "surfboard", "wave"],
        "bear": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "paw", "tail"],
        "bed": ["pillow", "room"],
        "bench": ["seat"],
        "bike": ["engine", "handle", "motorcycle", "seat", "street", "tire", "vehicle", "wheel"],
        "bird": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail", "wing"],
        "board": ["sign", "skateboard", "surfboard"],
        "boat": ["beach", "engine", "seat", "vehicle"],
        "book": ["letter", "logo", "number"],
        "boot": ["boy", "child", "girl", "guy", "kid", "lady", "leg", "man", "men", "people", "person", "player", "shoe", "skier", "woman"],
        "bottle": ["cap"],
        "bowl": [],
        "box": ["paper"],
        "boy": ["arm", "bag", "boot", "cap", "child", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "kid", "leg", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie", "umbrella"],
        "branch": ["leaf", "plant", "tree"],
        "building": ["door", "flag", "house", "railing", "roof", "room", "street", "tower", "window"],
        "bus": ["door", "engine", "letter", "logo", "number", "roof", "seat", "street", "tire", "vehicle", "wheel", "window", "windshield"],
        "cabinet": ["door", "drawer", "handle", "room", "shelf"],
        "cap": ["bottle", "boy", "child", "girl", "guy", "hair", "hat", "head", "kid", "lady", "letter", "logo", "man", "men", "number", "people", "person", "player", "skier", "woman"],
        "car": ["door", "engine", "handle", "letter", "logo", "number", "roof", "seat", "street", "tire", "trunk", "vehicle", "wheel", "window", "windshield"],
        "cat": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "paw", "tail"],
        "chair": ["room", "seat"],
        "child": ["arm", "bag", "boot", "boy", "cap", "coat", "ear", "eye", "face", "finger", "girl", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "kid", "leg", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie"],
        "clock": ["number", "room"],
        "coat": ["arm", "boy", "child", "girl", "guy", "jacket", "kid", "lady", "man", "men", "people", "person", "player", "skier", "tie", "woman"],
        "counter": ["room", "sink", "table"],
        "cow": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail"],
        "cup": ["glass"],
        "curtain": ["room", "window"],
        "desk": ["room", "table"],
        "dog": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "paw", "tail"],
        "door": ["building", "bus", "cabinet", "car", "handle", "house", "room", "train", "truck", "vehicle"],
        "drawer": ["cabinet", "handle", "room"],
        "ear": ["animal", "bear", "bird", "boy", "cat", "child", "cow", "dog", "elephant", "face", "giraffe", "girl", "guy", "head", "horse", "kid", "lady", "man", "men", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "elephant": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail", "trunk"],
        "engine": ["airplane", "bike", "boat", "bus", "car", "motorcycle", "plane", "train", "truck", "vehicle"],
        "eye": ["animal", "bear", "bird", "boy", "cat", "child", "cow", "dog", "elephant", "face", "giraffe", "girl", "guy", "head", "horse", "kid", "lady", "man", "men", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "face": ["animal", "bear", "bird", "boy", "cat", "child", "cow", "dog", "ear", "elephant", "eye", "giraffe", "girl", "guy", "head", "horse", "kid", "lady", "man", "men", "mouth", "nose", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "fence": ["post", "railing", "street"],
        "finger": ["arm", "boy", "child", "girl", "glove", "guy", "hand", "kid", "lady", "man", "men", "people", "person", "player", "skier", "woman"],
        "flag": ["building", "pole", "post"],
        "flower": ["leaf", "plant", "pot", "vase"],
        "food": ["banana", "fruit", "orange", "pizza", "vegetable"],
        "fork": [],
        "fruit": ["banana", "food", "orange"],
        "giraffe": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail"],
        "girl": ["arm", "bag", "boot", "cap", "child", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "kid", "leg", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie", "umbrella"],
        "glass": ["cup"],
        "glove": ["arm", "boy", "child", "finger", "girl", "guy", "hand", "kid", "lady", "man", "men", "people", "person", "player", "skier", "woman"],
        "guy": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "man", "men", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie"],
        "hair": ["boy", "cap", "child", "girl", "guy", "hat", "head", "helmet", "kid", "lady", "man", "men", "people", "person", "player", "skier", "woman"],
        "hand": ["arm", "boy", "child", "finger", "girl", "glove", "guy", "kid", "lady", "man", "men", "people", "person", "player", "skier", "woman"],
        "handle": ["bag", "basket", "bike", "cabinet", "car", "door", "drawer", "motorcycle", "truck", "umbrella"],
        "hat": ["boy", "cap", "child", "girl", "guy", "hair", "head", "kid", "lady", "letter", "logo", "man", "men", "number", "people", "person", "player", "skier", "woman"],
        "head": ["animal", "bear", "bird", "boy", "cap", "cat", "child", "cow", "dog", "ear", "elephant", "eye", "face", "giraffe", "girl", "guy", "hair", "hat", "helmet", "horse", "kid", "lady", "man", "men", "mouth", "neck", "nose", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "helmet": ["boy", "child", "girl", "guy", "hair", "head", "kid", "lady", "letter", "logo", "man", "men", "number", "people", "person", "player", "skier", "woman"],
        "hill": ["mountain", "rock", "tree"],
        "horse": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail"],
        "house": ["building", "door", "roof", "room", "window"],
        "jacket": ["arm", "boy", "child", "coat", "girl", "guy", "kid", "lady", "man", "men", "people", "person", "player", "skier", "tie", "woman"],
        "jean": ["boy", "child", "girl", "guy", "kid", "lady", "leg", "man", "men", "pant", "people", "person", "player", "skier", "woman"],
        "kid": ["arm", "bag", "boot", "boy", "cap", "child", "coat", "ear", "eye", "face", "finger", "girl", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie"],
        "kite": [],
        "lady": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie", "woman"],
        "lamp": ["light", "room", "stand"],
        "laptop": ["room", "screen"],
        "leaf": ["branch", "flower", "plant", "tree"],
        "leg": ["animal", "bear", "bird", "boot", "boy", "cat", "child", "cow", "dog", "elephant", "giraffe", "girl", "guy", "horse", "jean", "kid", "lady", "man", "men", "pant", "people", "person", "player", "sheep", "shoe", "short", "skier", "sneaker", "sock", "woman", "zebra"],
        "letter": ["airplane", "bag", "book", "bus", "cap", "car", "hat", "helmet", "paper", "plane", "screen", "shirt", "sign", "train", "truck", "vehicle"],
        "light": ["lamp", "pole", "post", "street"],
        "logo": ["airplane", "bag", "book", "bus", "cap", "car", "hat", "helmet", "paper", "plane", "screen", "shirt", "sign", "train", "truck", "vehicle"],
        "man": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "guy", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "men", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie", "umbrella"],
        "men": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "guy", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "man", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie"],
        "motorcycle": ["bike", "engine", "handle", "seat", "street", "tire", "vehicle", "wheel"],
        "mountain": ["hill", "rock", "ski", "skier", "snow", "tree"],
        "mouth": ["animal", "bear", "bird", "boy", "cat", "child", "cow", "dog", "elephant", "face", "giraffe", "girl", "guy", "head", "horse", "kid", "lady", "man", "men", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "neck": ["animal", "bear", "bird", "boy", "cat", "child", "cow", "dog", "elephant", "giraffe", "girl", "guy", "head", "horse", "kid", "lady", "man", "men", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "nose": ["animal", "bear", "bird", "boy", "cat", "child", "cow", "dog", "elephant", "face", "giraffe", "girl", "guy", "head", "horse", "kid", "lady", "man", "men", "people", "person", "player", "sheep", "skier", "woman", "zebra"],
        "number": ["airplane", "bag", "book", "bus", "cap", "car", "clock", "hat", "helmet", "paper", "plane", "screen", "shirt", "sign", "train", "truck", "vehicle"],
        "orange": ["food", "fruit"],
        "pant": ["boy", "child", "girl", "guy", "jean", "kid", "lady", "leg", "man", "men", "people", "person", "player", "short", "skier", "woman"],
        "paper": ["box", "letter", "logo", "number"],
        "paw": ["animal", "bear", "cat", "dog"],
        "people": ["arm", "bag", "beach", "boot", "boy", "cap", "child", "coat", "ear", "eye", "face", "finger", "girl", "glove", "guy", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "kid", "lady", "leg", "man", "men", "mouth", "neck", "nose", "pant", "person", "player", "shirt", "shoe", "short", "skier", "sneaker", "sock", "street", "tie", "towel", "umbrella", "woman"],
        "person": ["arm", "bag", "beach", "boot", "boy", "cap", "child", "coat", "ear", "eye", "face", "finger", "girl", "glove", "guy", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "kid", "lady", "leg", "man", "men", "mouth", "neck", "nose", "pant", "people", "player", "shirt", "shoe", "short", "skier", "sneaker", "sock", "street", "tie", "towel", "umbrella", "woman"],
        "phone": ["screen"],
        "pillow": ["bed", "room"],
        "pizza": ["food"],
        "plane": ["airplane", "engine", "letter", "logo", "number", "tire", "vehicle", "wheel", "window", "wing"],
        "plant": ["branch", "flower", "leaf", "pot", "tree", "vase"],
        "plate": [],
        "player": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "mouth", "neck", "nose", "pant", "people", "person", "racket", "shirt", "shoe", "short", "skateboard", "ski", "skier", "sneaker", "sock", "surfboard", "tie"],
        "pole": ["flag", "light", "post", "sign", "street", "umbrella", "wire"],
        "post": ["fence", "flag", "light", "pole", "sign", "street", "wire"],
        "pot": ["flower", "plant"],
        "racket": ["player"],
        "railing": ["building", "fence"],
        "rock": ["hill", "mountain", "snow"],
        "roof": ["building", "bus", "car", "house", "truck", "vehicle"],
        "room": ["bed", "building", "cabinet", "chair", "clock", "counter", "curtain", "desk", "door", "drawer", "house", "lamp", "laptop", "pillow", "screen", "seat", "shelf", "sink", "table", "tile", "toilet", "window"],
        "screen": ["laptop", "letter", "logo", "number", "phone", "room", "stand"],
        "seat": ["bench", "bike", "boat", "bus", "car", "chair", "motorcycle", "room", "train", "truck", "vehicle"],
        "sheep": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail"],
        "shelf": ["cabinet", "room"],
        "shirt": ["arm", "boy", "child", "girl", "guy", "kid", "lady", "letter", "logo", "man", "men", "number", "people", "person", "player", "skier", "tie", "woman"],
        "shoe": ["boot", "boy", "child", "girl", "guy", "kid", "lady", "leg", "man", "men", "people", "person", "player", "skier", "sneaker", "woman"],
        "short": ["boy", "child", "girl", "guy", "kid", "lady", "leg", "man", "men", "pant", "people", "person", "player", "skier", "woman"],
        "sidewalk": ["sign", "street"],
        "sign": ["board", "letter", "logo", "number", "pole", "post", "sidewalk", "stand", "street"],
        "sink": ["counter", "room"],
        "skateboard": ["board", "player"],
        "ski": ["mountain", "player", "skier", "snow", "track"],
        "skier": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "leg", "mountain", "mouth", "neck", "nose", "pant", "people", "person", "player", "shirt", "shoe", "short", "ski", "sneaker", "snow", "sock", "tie"],
        "sneaker": ["boy", "child", "girl", "guy", "kid", "lady", "leg", "man", "men", "people", "person", "player", "shoe", "skier", "woman"],
        "snow": ["mountain", "rock", "ski", "skier"],
        "sock": ["boy", "child", "girl", "guy", "kid", "lady", "leg", "man", "men", "people", "person", "player", "skier", "woman"],
        "stand": ["lamp", "screen", "sign"],
        "street": ["bike", "building", "bus", "car", "fence", "light", "motorcycle", "people", "person", "pole", "post", "sidewalk", "sign", "truck", "vehicle"],
        "surfboard": ["beach", "board", "player"],
        "table": ["counter", "desk", "room"],
        "tail": ["animal", "bear", "bird", "cat", "cow", "dog", "elephant", "giraffe", "horse", "sheep", "wing", "zebra"],
        "tie": ["boy", "child", "coat", "girl", "guy", "jacket", "kid", "lady", "man", "men", "people", "person", "player", "shirt", "skier", "woman"],
        "tile": ["room"],
        "tire": ["airplane", "bike", "bus", "car", "motorcycle", "plane", "truck", "vehicle", "wheel"],
        "toilet": ["room"],
        "towel": ["people", "person"],
        "tower": ["building"],
        "track": ["ski", "train"],
        "train": ["door", "engine", "letter", "logo", "number", "seat", "track", "vehicle", "wheel", "window"],
        "tree": ["branch", "hill", "leaf", "mountain", "plant", "trunk"],
        "truck": ["door", "engine", "handle", "letter", "logo", "number", "roof", "seat", "street", "tire", "trunk", "vehicle", "wheel", "window", "windshield"],
        "trunk": ["animal", "car", "elephant", "tree", "truck", "vehicle"],
        "umbrella": ["boy", "girl", "handle", "man", "people", "person", "pole", "woman"],
        "vase": ["flower", "plant"],
        "vegetable": ["food"],
        "vehicle": ["airplane", "bike", "boat", "bus", "car", "door", "engine", "letter", "logo", "motorcycle", "number", "plane", "roof", "seat", "street", "tire", "train", "truck", "trunk", "wheel", "window", "windshield"],
        "wave": ["beach"],
        "wheel": ["airplane", "bike", "bus", "car", "motorcycle", "plane", "tire", "train", "truck", "vehicle"],
        "window": ["airplane", "building", "bus", "car", "curtain", "house", "plane", "room", "train", "truck", "vehicle", "windshield"],
        "windshield": ["bus", "car", "truck", "vehicle", "window"],
        "wing": ["airplane", "animal", "bird", "plane", "tail"],
        "wire": ["pole", "post"],
        "woman": ["arm", "bag", "boot", "cap", "coat", "ear", "eye", "face", "finger", "glove", "hair", "hand", "hat", "head", "helmet", "jacket", "jean", "lady", "leg", "mouth", "neck", "nose", "pant", "people", "person", "shirt", "shoe", "short", "sneaker", "sock", "tie", "umbrella"],
        "zebra": ["animal", "ear", "eye", "face", "head", "leg", "mouth", "neck", "nose", "tail"],
    }

    supported_priors = [NUS_COMPATIBILITY, COCO_COMPATIBILITY, VOC_COMPATIBILITY, VG_COMPATIBILITY]
    for compatibility in supported_priors:
        if set(compatibility.keys()) == set(classnames):
            compatibility_mat = torch.zeros((len(classnames), len(classnames)), dtype=torch.bool)
            for name_i, compatibility_names in compatibility.items():
                i = classnames.index(name_i)
                for name_j in compatibility_names:
                    j = classnames.index(name_j)
                    compatibility_mat[i, j] = True
                compatibility_mat[i, i] = False
            return compatibility_mat
    raise ValueError(f"Unsupported classnames for compatibility matrix: {classnames}")


def random_compatibility_matrix(classnames, prob=0.05):
    num_classes = len(classnames)
    compatibility_mat = torch.zeros((num_classes, num_classes), dtype=torch.bool)
    noise = torch.rand((num_classes, num_classes))
    compatibility_mat[noise < prob] = True
    compatibility_mat.fill_diagonal_(False)
    return compatibility_mat


class LoCoModel(nn.Module):
    """Frozen LoCo inference with optional pretrained TaI global predictions."""

    def __init__(
        self,
        clip_model: CLIP,
        classnames: list[str],
        multi_scale: list[int] = [1, 2, 3],
        alpha: float = 0.0,
        scoring_rule: Literal["selective-softmax", "softmax", "logits"] = "selective-softmax",
        visual_encoder: Literal["scclip", "clip", "clearclip"] = "scclip",
        global_crop_reduction: Literal["mean", "max", "none"] = "none",
        beta: float = 1.0,
        temperature: float = 1 / 40.0,
        compatibility_prior: Literal["random", "llm"] = "llm",
    ):
        super().__init__()
        self.visual_encoder = DenseSCClipEncoder(clip_model)
        self.classnames = classnames
        self.grid_cropper = GridCropper(
            scale_level=multi_scale,
            overlap_ratio=0.5,
            patch_size=(224, 224),
        )
        self.clip_features = nn.Buffer(torch.zeros(0))
        self.tai_features = nn.Buffer(torch.zeros(0))
        self.alpha = alpha
        self.scoring_rule = scoring_rule
        # Each row includes the target class and its incompatible competitors.
        self.normalization_mask = nn.Buffer(~(compatibility_matrix(classnames) if compatibility_prior == "llm" else random_compatibility_matrix(classnames, prob=0.05)))
        self.beta = beta
        self.temperature = temperature

        self.global_crop_reduction = global_crop_reduction

        if visual_encoder == "clip":
            self.visual_encoder = DenseVisualEncoder(clip_model=clip_model)

        if visual_encoder == "clearclip":
            self.visual_encoder.module.anomaly_resolve = False
            self.visual_encoder.module.multi_layer_fuse = False
            self.visual_encoder.module.attn_enhance = False
            self.visual_encoder.module.feat_aggregate = False
            self.visual_encoder.module.cls_token_fusing_factor = 0

    def train(self, mode: bool = True):
        super().train(False)
        return self


    def learnable_parameters(self):
        return []


    def compute_crop_logits(self, image: Tensor) -> Tensor:
        # img_feats_global: [B, D], img_feats_local: [B, H, W, D]
        img_feats_global, img_feats_local = self.visual_encoder(image, normalize=True, spatialize=True)

        # Pure LoCo does not require pretrained TaI prototypes.
        global_logits = (
            img_feats_global @ self.tai_features.T
            if self.alpha != 0
            else img_feats_global.new_zeros((image.shape[0], len(self.classnames)))
        )  # [B, C]

        local_logits = img_feats_local @ self.clip_features.T  # [B, H, W, C]

        return global_logits, local_logits  # [B, C], [B, H, W, C]

    def selective_softmax(self, local_logits: Tensor) -> Tensor:
        B, H, W, C = local_logits.shape
        local_logits = local_logits[:, :, :, None, :].expand(B, H, W, C, C)
        masked_logits = local_logits.masked_fill(
            (~self.normalization_mask.data)[None, None, None, :, :],
            float("-inf"),
        )  # [B, H, W, C, C]
        return (masked_logits / self.temperature).softmax(dim=-1).diagonal(dim1=-2, dim2=-1)

    def local_prediction(self, local_logits: Tensor) -> Tensor:
        if self.scoring_rule == "selective-softmax":
            local_scores = self.selective_softmax(local_logits)
        elif self.scoring_rule == "softmax":
            local_scores = (local_logits / self.temperature).softmax(dim=-1)
        elif self.scoring_rule == "logits":
            local_scores = local_logits
        return local_scores.amax(dim=[1, 2])  # [B, C]
    
    def forward_multiscale(self, image: Tensor) -> Tensor:
        crop_images, crop_boxes = self.grid_cropper.crop(image)
        K, B, _, H1, W1 = crop_images.shape

        crop_images = crop_images.view(-1, 3, H1, W1)  # [K*B, 3, H1, W1]
        crop_global_logits, crop_local_logits = self.compute_crop_logits(crop_images)

        crop_global_logits = crop_global_logits.view(K, B, *crop_global_logits.shape[1:])  # [K, B, C]
        crop_local_logits = crop_local_logits.view(K, B, *crop_local_logits.shape[1:])  # [K, B, H2, W2, C]

        if self.global_crop_reduction == "mean":
            global_logits = crop_global_logits.mean(dim=0)  # [B, C]
        elif self.global_crop_reduction == "max":
            global_logits = crop_global_logits.amax(dim=0)  # [B, C]
        elif self.global_crop_reduction == "none":
            global_logits = crop_global_logits[0]  # [B, C]
        local_logits = self.grid_cropper.merge(crop_local_logits, crop_boxes, target_size=(28, 28)) # [B, H3, W3, C]

        return global_logits, local_logits

    def forward(self, image: Tensor, debug: bool = False) -> Tensor:
        global_logits, local_logits = self.forward_multiscale(image)

        if debug:
            return global_logits, local_logits

        global_scores = global_logits
        local_scores = self.local_prediction(local_logits)
        scores = torch.clamp(self.alpha * global_scores + self.beta * local_scores, min=0, max=1)

        return scores


class LoCoConfig(BaseConfig):
    backbone: Literal["ViT-B/16"] = "ViT-B/16"
    image_size: int = 448
    dataset: str = "coco2014"
    downsample_ratio: float = 0.1
    batch_size: int = 4
    max_epochs: int = 1

    proto_dir: str = str(Path(__file__).resolve().parent / "proto")
    dataset_root: str = str(Path.home() / "dataset")

    scoring_rule: Literal["selective-softmax", "softmax", "logits"] = "selective-softmax"
    visual_encoder: Literal["scclip", "clip", "clearclip"] = "scclip"

    alpha: float = 0.0
    beta: float = 1.0
    temperature: float = 1 / 40.0
    multi_scale: list[int] = [1, 2, 3]
    global_crop_reduction: Literal["mean", "max", "none"] = "none"
    compatibility_prior: Literal["random", "llm"] = "llm"


class LoCoTrainer(BaseTrainer):

    def create_datasets(self, cfg):
        transform = t.Compose([t.Resize((cfg.image_size, cfg.image_size)), t.ToTensor(), t.Normalize()])
        test_dataset = create_dataset(
            cfg.dataset,
            divide="test",
            base=str(Path(cfg.dataset_root).expanduser()),
            downsample_ratio=cfg.downsample_ratio,
            add_index=True,
            transform=transform,
        )
        # BaseTrainer expects three slots even for inference-only execution.
        return test_dataset, test_dataset, test_dataset

    def create_loaders(self, train_dataset, valid_dataset, test_dataset, cfg):
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=len(test_dataset) < 10000,
        )
        return test_loader, test_loader, test_loader


    def create_model(self, dataset, cfg):
        classnames = list(dataset.category_name.values())

        clip_model, _ = create_clip(cfg.backbone)
        model = LoCoModel(
            clip_model=clip_model,
            classnames=classnames,
            multi_scale=cfg.multi_scale,
            alpha=cfg.alpha,
            scoring_rule=cfg.scoring_rule,
            visual_encoder=cfg.visual_encoder,
            beta=cfg.beta,
            global_crop_reduction=cfg.global_crop_reduction,
            temperature=cfg.temperature,
            compatibility_prior=cfg.compatibility_prior,
        )
        model.requires_grad_(False)

        proto_dir = Path(cfg.proto_dir).expanduser()
        clip_ckpt = torch.load(
            proto_dir / f"clip-vitb16-templated-{cfg.dataset}.pt",
            map_location=self.device,
            weights_only=True,
        )
        model.clip_features.data = F.normalize(clip_ckpt["global"], dim=-1)
        if cfg.alpha != 0:
            tai_ckpt = torch.load(
                proto_dir / f"taidpt-vitb16-{cfg.dataset}.pt",
                map_location=self.device,
                weights_only=True,
            )
            model.tai_features.data = F.normalize(tai_ckpt["global"], dim=-1)

        return model


if __name__ == "__main__":
    entry(config=LoCoConfig, trainer=LoCoTrainer, train=False, test=True)
