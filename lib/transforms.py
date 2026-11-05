import torch
import numpy as np
from torchvision import transforms as t
from torch import Tensor
from PIL import Image, ImageDraw
import random
import numpy as np
from typing import List
from collections.abc import Sequence
from torchvision.transforms import functional as F
from torchvision.transforms import (
    AugMix,
    AutoAugment,
    AutoAugmentPolicy,
    CenterCrop,
    ColorJitter,
    Compose,
    ConvertImageDtype,
    ElasticTransform,
    FiveCrop,
    GaussianBlur,
    Grayscale,
    InterpolationMode,
    Lambda,
    LinearTransformation,
    # Normalize,
    PILToTensor,
    Pad,
    RandAugment,
    RandomAdjustSharpness,
    RandomAffine,
    RandomApply,
    RandomAutocontrast,
    RandomChoice,
    RandomCrop,
    RandomEqualize,
    RandomErasing,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomInvert,
    RandomOrder,
    RandomPerspective,
    RandomPosterize,
    RandomResizedCrop,
    RandomRotation,
    RandomSolarize,
    RandomVerticalFlip,
    # Resize,
    TenCrop,
    # ToPILImage,
    # ToTensor,
    TrivialAugmentWide,
)
from .backbone.clip import tokenize

__all__ = [
    "AugMix",
    "AutoAugment",
    "AutoAugmentPolicy",
    "CenterCrop",
    "ColorJitter",
    "Compose",
    "ConvertImageDtype",
    "ElasticTransform",
    "FiveCrop",
    "GaussianBlur",
    "Grayscale",
    "InterpolationMode",
    "Lambda",
    "LinearTransformation",
    "PILToTensor",
    "Pad",
    "RandAugment",
    "RandomAdjustSharpness",
    "RandomAffine",
    "RandomApply",
    "RandomAutocontrast",
    "RandomChoice",
    "RandomCrop",
    "RandomEqualize",
    "RandomErasing",
    "RandomGrayscale",
    "RandomHorizontalFlip",
    "RandomInvert",
    "RandomOrder",
    "RandomPerspective",
    "RandomPosterize",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomSolarize",
    "RandomVerticalFlip",
    "Resize",
    "TenCrop",
    "ToPILImage",
    "ToTensor",
    "TrivialAugmentWide",
    "Identity",
    "Normalize",
    "UnNormalize",
    "AlignedResize",
    "RandomCutout",
    "PatchCrop",
    "Parallel",
    "UnParallel",
    "Tokenize",
    "create_transform",
]


IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


class Identity(t.Lambda):
    def __init__(self):
        super().__init__(lambda x: x)

class Resize(t.Resize):
    def __init__(self, size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True):
        super().__init__(size, interpolation, max_size, antialias)


class Normalize(t.Normalize):
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
        super().__init__(mean, std, inplace)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(tensor, Tensor):
            raise TypeError(f'Normalize() take torch.Tensor as input, got {type(tensor)}')

        return super().forward(tensor)


class UnNormalize(torch.nn.Module):
    """Un-normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will un-normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = input[channel] * std[channel] + mean[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = list(-torch.tensor(self.mean) / torch.tensor(self.std))
        std = list(1 / torch.tensor(self.std))
        # std' = 1 / std, mean' = -mean / std
        return t.functional.normalize(tensor, mean, std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class AlignedResize(torch.nn.Module):
    def __init__(self, tile_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None):
        super().__init__()
        self.tile_size = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size  # (h, w)
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(img, (Image.Image, torch.Tensor)):
            (tile_h, tile_w), (_, orig_h, orig_w) = self.tile_size, F.get_dimensions(img)
            h, w = [(int(np.ceil(dim / tile_dim) * tile_dim)) for dim, tile_dim in zip((orig_h, orig_w), (tile_h, tile_w))]
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        return F.resize(img, (h, w), self.interpolation, self.max_size, self.antialias)

    def __repr__(self) -> str:
        detail = (
            f"(size={self.tile_size}, interpolation={self.interpolation.value},"
            f"max_size={self.max_size}, antialias={self.antialias})"
        )
        return f"{self.__class__.__name__}{detail}"


class RandomCutout(torch.nn.Module):
    """Cutout the given image with a random color rectangle mask.

    Args:
        cutout_factor (float, between 0 and 1): Control the size of the mask.
        inplace (bool, optional): Bool to make this operation in-place.
    """

    def __init__(self, cutout_factor=0.5, inplace=False):
        super().__init__()
        self.cutout_factor = cutout_factor
        self.inplace = inplace

    def forward(self, image):
        """
        Args:
            image (PIL.Image or torch.Tensor): Input image.

        Returns:
            PIL.Image or torch.Tensor: Image with random cutout applied.
        """
        if isinstance(image, Image.Image):
            return self._cutout_pil(image)
        elif isinstance(image, torch.Tensor):
            return self._cutout_tensor(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _cutout_pil(self, image):
        """Apply cutout to a PIL.Image."""
        if not self.inplace:
            image = image.copy()

        img_draw = ImageDraw.Draw(image)
        w, h = image.size
        y1, y2, x1, x2 = self._get_cutout_coordinates(h, w)
        fill_color = self._get_random_color_pil()
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return image

    def _cutout_tensor(self, image):
        """Apply cutout to a torch.Tensor."""
        if not self.inplace:
            image = image.clone()

        _, h, w = image.shape
        y1, y2, x1, x2 = self._get_cutout_coordinates(h, w)
        fill_color = self._get_random_color_tensor(image.device)
        image[:, y1:y2, x1:x2] = fill_color

        return image

    def _get_cutout_coordinates(self, height, width):
        """Calculate the coordinates for the cutout rectangle."""
        h_cutout = int(self.cutout_factor * height)
        w_cutout = int(self.cutout_factor * width)
        y_c = np.random.randint(height)
        x_c = np.random.randint(width)

        y1 = np.clip(y_c - h_cutout // 2, 0, height)
        y2 = np.clip(y_c + h_cutout // 2, 0, height)
        x1 = np.clip(x_c - w_cutout // 2, 0, width)
        x2 = np.clip(x_c + w_cutout // 2, 0, width)

        return y1, y2, x1, x2

    def _get_random_color_pil(self):
        """Generate a random RGB color for PIL.Image."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _get_random_color_tensor(self, device):
        """Generate a random RGB color tensor for torch.Tensor."""
        return torch.tensor(
            [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)],
            device=device,
        ).view(3, 1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutout_factor={self.cutout_factor}, inplace={self.inplace})"


class PatchCrop(torch.nn.Module):
    """
    Crop an image into a grid of patches of the given size.

    Args:
        size (int or tuple): Size of each patch (height, width).
    """

    def __init__(self, size=None, nums=None):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size  # (patch_size_y, patch_size_x)
        self.nums = (nums, nums) if isinstance(nums, int) else nums  # (num_patches_y, num_patches_x)
        assert (self.size is None) != (self.nums is None)

    def forward(self, image):
        """
        Args:
            image (PIL.Image or torch.Tensor): Input image.

        Returns:
            List[PIL.Image] or List[torch.Tensor]: List of patches.
        """
        if isinstance(image, Image.Image):
            return self._crop_pil(image)
        elif isinstance(image, torch.Tensor):
            return self._crop_tensor(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _crop_pil(self, image):
        """Crop a PIL.Image into patches."""
        img_width, img_height = image.size
        patches = []

        # Calculate the number of patches in height and width
        num_patches_y = img_height // self.size[0] if self.nums is None else self.nums[0]
        num_patches_x = img_width // self.size[1] if self.nums is None else self.nums[1]
        patch_size_y = self.size[0] if self.nums is None else img_height // self.nums[0]
        patch_size_x = self.size[1] if self.nums is None else img_width // self.nums[1]

        # Crop the image into patches
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                left = j * patch_size_x
                top = i * patch_size_y
                right = left + patch_size_x
                bottom = top + patch_size_y
                patch = image.crop((left, top, right, bottom))
                patches.append(patch)

        return patches

    def _crop_tensor(self, image):
        """Crop a torch.Tensor into patches."""
        _, img_height, img_width = image.shape
        patches = []

        # Calculate the number of patches in height and width
        num_patches_y = img_height // self.size[0] if self.nums is None else self.nums[0]
        num_patches_x = img_width // self.size[1] if self.nums is None else self.nums[1]
        patch_size_y = self.size[0] if self.nums is None else img_height // self.nums[0]
        patch_size_x = self.size[1] if self.nums is None else img_width // self.nums[1]

        # Crop the image into patches
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                top = i * patch_size_y
                left = j * patch_size_x
                patch = image[:, top : top + patch_size_y, left : left + patch_size_x]
                patches.append(patch)

        return torch.stack(patches)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class Parallel(torch.nn.Module):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        imgs = [t(img) for t in self.transforms]
        return imgs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class UnParallel(torch.nn.Module):
    """
    Added after a Parallel to make transform be applied on the list of images

        Args:
            transforms (PIL Image): 输入图片。

        Returns:
            tuple: 包含两个增强后的图片 (img1, img2)。
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for trans in self.transforms:
            imgs = apply_transform_recursive(imgs, trans)
        return imgs

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for trans in self.transforms:
            format_string += "\n"
            format_string += f"    {trans}"
        format_string += "\n)"
        return format_string


class Stack(torch.nn.Module):
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, imgs):
        return torch.stack(imgs, dim=self.dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


class ToPILImage(t.ToPILImage):
    """Convert a tensor or an ndarray to PIL Image. If it is PIL Image already, ignore it."""

    def __call__(self, pic):
        if isinstance(pic, Image.Image):
            return pic
        else:
            return super().__call__(pic)


class ToTensor(t.ToTensor):
    """Convert a PIL Image or numpy.ndarray to tensor. If it is tensor already, ignore it."""

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        else:
            return super().__call__(pic)


class Tokenize(torch.nn.Module):
    def __call__(self, caption: str | Tensor) -> Tensor:
        if isinstance(caption, str):
            return tokenize(caption)[0]  # [77]
        else:
            return caption


def apply_transform_recursive(data, transform):
    if isinstance(data, list):
        return [apply_transform_recursive(item, transform) for item in data]
    else:
        return transform(data)


def create_transform(names: List[str], image_size: int = 224):
    SUPPORTED_TRANSFORM = {
        "Normalize": Normalize(),
        "UnNormalize": UnNormalize(),
        "Resize": t.Resize((image_size, image_size)),
        "RandomHorizontalFlip": t.RandomHorizontalFlip(),
        "ToTensor": t.ToTensor(),
        "RandomCutout": RandomCutout(),
        "RandomResizedCrop": t.RandomResizedCrop((image_size, image_size)),
        "RandAugment": t.RandAugment(),
        "AutoAugment": t.AutoAugment(),
    }

    return t.Compose([SUPPORTED_TRANSFORM[name] for name in names])
