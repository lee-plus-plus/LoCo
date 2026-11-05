from torch import nn, Tensor
from typing import Literal


def ranking_loss(
    logit: Tensor,
    target: Tensor,
    scale: float = 2.0,
    margin: float = 1.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    # logit: [batch_size, num_classes] in (-inf, +inf)
    # target: [batch_size, num_classes] in {0, 1}
    logit = logit * scale
    target = target.float()

    # for y_ij = 0 and y_ik = 1, ((p_ij - p_ik) + margin) will be punished if it is > 0,
    # i.e., p_ij should be < p_ik with margin
    pairwise_logit = logit[:, :, None] - logit[:, None, :] + margin
    mask = (pairwise_logit > 0).float() * ((1 - target[:, :, None]) * target[:, None, :])
    loss = mask * pairwise_logit  # [batch_size, num_classes, num_classes]
    loss = loss.sum(dim=[1, 2])

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    return loss


class RankingLoss(nn.Module):
    def __init__(
        self,
        scale: float = 2.0,
        margin: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.reduction = reduction

    def forward(self, input, target, reduction=None) -> Tensor:
        """Parameters
        -------------
        input: (batch_size, num_classes)
        target: (batch_size, num_classes)
        """
        return ranking_loss(input, target, reduction=(reduction or self.reduction))
