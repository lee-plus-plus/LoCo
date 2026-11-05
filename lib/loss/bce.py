import torch
from torch import nn, Tensor
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from typing import Literal


def bce_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    loss = binary_cross_entropy(input.float(), target.float(), weight=weight, reduction="none")

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    return loss


def bce_loss_with_logits(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    loss = binary_cross_entropy_with_logits(input.float(), target.float(), weight=weight, reduction="none")

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    return loss


class BceLoss(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tensor:
        """Parameters
        ----------
        input: (batch_size, num_classes)
        target: (batch_size, num_classes)
        """
        return bce_loss(
            input,
            target,
            weight=weight,
            reduction=(reduction or self.reduction),
        )


class BceWithLogitsLoss(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tensor:
        """Parameters
        ----------
        input: (batch_size, num_classes)
        target: (batch_size, num_classes)
        """
        return bce_loss_with_logits(
            input,
            target,
            weight=weight,
            reduction=(reduction or self.reduction),
        )
