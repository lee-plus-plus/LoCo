"""ASL

Ridnik, Tal, et al.
"Asymmetric loss for multi-label classification."
Proceedings of the IEEE/CVF international conference on computer vision. 2021.

recommended setting in source paper: gamma_neg=4, gamma_pos=0, clip=0.05
"""

import torch
from torch import nn, Tensor
from typing import Literal


# Asymmetric Loss For Multi-Label Classification
def asymmetric_bce_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    no_focal_grad: bool = True,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    input_pos = input
    input_neg = 1 - input

    # Probability Shifting for Negative Prediction
    input_neg = (input_neg + clip).clamp(max=1)

    # Binary Cross Entropy
    los_pos = -target * torch.log(input_pos.clamp(min=eps))
    los_neg = -(1 - target) * torch.log(input_neg.clamp(min=eps))
    loss = los_pos + los_neg  # (batch_size, num_classes)

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        base = input_pos * target + input_neg * (1 - target)
        gamma = gamma_pos * target + gamma_neg * (1 - target)
        focusing_weight = torch.pow(1 - base, gamma)

        if no_focal_grad:
            focusing_weight.detach_()

        loss *= focusing_weight

    if weight is not None:
        loss *= weight

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    return loss


def asymmetric_bce_loss_with_logits(input, target, *args, **kwargs):
    input = torch.sigmoid(input)
    return asymmetric_bce_loss(input, target, *args, **kwargs)


class AsymmetricBceLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        no_focal_grad: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.no_focal_grad = no_focal_grad
        self.reduction = reduction

    def forward(self, input, target, weight=None, reduction=None):
        """Parameters
        ----------
        input: (batch_size, num_classes)
        target: (batch_size, num_classes)

        loss_pos = ((1 - input_pos) ** gamma_pos) * log(input_pos)
        loss_neg = ((1 - input_neg) ** gamma_neg) * log(input_neg)
        loss = target * loss_pos + (1 - target) * loss_neg
        """
        return asymmetric_bce_loss(
            input,
            target,
            weight=weight,
            gamma_neg=self.gamma_neg,
            gamma_pos=self.gamma_pos,
            clip=self.clip,
            eps=self.eps,
            no_focal_grad=self.no_focal_grad,
            reduction=(reduction or self.reduction),
        )


class AsymmetricBceWithLogitsLoss(AsymmetricBceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target, *args, **kwargs):
        input = torch.sigmoid(input)
        return super().forward(input, target, *args, **kwargs)
