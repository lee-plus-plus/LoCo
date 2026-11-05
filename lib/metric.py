from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.functional.classification import multilabel_f1_score
import warnings

from torchmetrics.classification import (
    MultilabelHammingDistance,
    # MultilabelRankingLoss,  # TODO: unsafe
    # MultilabelAveragePrecision,
    MultilabelAccuracy,
    MultilabelF1Score,
    MulticlassHammingDistance,
    # MulticlassRankingLoss,  # TODO: unsafe
    MulticlassAveragePrecision,  # TODO: unsafe
    MulticlassAccuracy,
    MulticlassF1Score,
)


@torch.no_grad()
def binary_average_precision_safe(
    preds: Tensor,
    target: Tensor,
    pos_label: int = 1,
) -> Tensor:
    """Calculate the TPs and false positives for all unique thresholds in the preds tensor.

    It will not cast preds into [0, 1] by sigmoid, so that large predition values' order is safely kept.
    """

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]
    desc_score_indices = torch.argsort(preds, descending=True)

    preds = preds[desc_score_indices]
    target = target[desc_score_indices]

    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target, dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    fps, tps, thresholds = fps, tps, preds[threshold_idxs]
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    if (preds == 0).all():  # all labels are negative, recall is undefined
        warnings.warn(
            "No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.",
            UserWarning,
        )
        recall = torch.ones_like(recall)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = torch.cat([precision.flip(0), torch.ones(1, dtype=precision.dtype, device=precision.device)])
    recall = torch.cat([recall.flip(0), torch.zeros(1, dtype=recall.dtype, device=recall.device)])
    thresholds = thresholds.flip(0).detach().clone()
    # return precision, recall, thresholds
    return -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])


def binary_f1_score(preds: Tensor, targets: Tensor, pos_label: int = 1) -> Tensor:
    y_true = targets.int()
    y_pred = preds.int()

    tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
    fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())

    return 2 * tp / (2 * tp + fp + fn + 1e-8)


@torch.no_grad
def binarize_topk(preds: Tensor, k: int, dim: int = -1) -> Tensor:
    """Binarize the predictions by keeping only the top-k values as 1 and the rest as 0."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > preds.size(dim):
        raise ValueError("k must be less than or equal to the number of classes.")
    
    topk_indices = torch.topk(preds, k=k, dim=dim).indices
    binarized_preds = torch.zeros_like(preds, dtype=torch.int)
    binarized_preds.scatter_(dim, topk_indices, 1)
    
    return binarized_preds

class MultilabelAveragePrecision(Metric):
    """
    sample-wise Multi-label Average Precision (instead of class-wise),
    equals to `sklearn.metrics.average_precision_score`
    """

    is_differentiable = None
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=torch.zeros((0, num_labels), dtype=float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros((0, num_labels), dtype=int), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self):
        ap_score = torch.tensor(
            [binary_average_precision_safe(preds, target) for preds, target in zip(self.preds.T, self.target.T)]
        )
        ap_score = ap_score[~ap_score.isnan()]
        return ap_score.mean()


class MultilabelAveragePrecision2(Metric):
    """
    sample-wise Multi-label Average Precision (instead of class-wise),
    equals to `sklearn.metrics.average_precision_score`
    """

    is_differentiable = None
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=torch.zeros((0, num_labels), dtype=float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros((0, num_labels), dtype=int), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self):
        ap_score = torch.tensor(
            [binary_average_precision_safe(preds, target) for preds, target in zip(self.preds, self.target)]
        )
        ap_score = ap_score[~ap_score.isnan()]
        return ap_score.mean()


class MultilabelTopKF1Score(Metric):
    """Computes the F1-score for the top-k predictions in a multi-label classification setting."""

    is_differentiable = None
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_labels: int, k: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.add_state("preds", default=torch.zeros((0, num_labels), dtype=float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros((0, num_labels), dtype=int), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = binarize_topk(preds, k=self.k, dim=-1)
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self) -> Tensor:
        f1_score = torch.tensor(
            [binary_f1_score(preds, target) for preds, target in zip(self.preds.T, self.target.T)]
        )
        f1_score = f1_score[~f1_score.isnan()]
        return f1_score.mean()


class MultilabelTopKF1Score2(MultilabelTopKF1Score):
    """Computes the F1-score for the top-k predictions in a multi-label classification setting."""

    is_differentiable = None
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_labels: int, k: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.add_state("preds", default=torch.zeros((0, num_labels), dtype=float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros((0, num_labels), dtype=int), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = binarize_topk(preds, k=self.k, dim=-1)
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self):
        f1_score = torch.tensor(
            [binary_f1_score(preds, target) for preds, target in zip(self.preds, self.target)]
        )
        f1_score = f1_score[~f1_score.isnan()]
        return f1_score.mean()


class MultilabelBestF1Score(Metric):
    """Multilabel F1 that selects the best threshold on accumulated data."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_labels: int,
        average: str | None = "macro",
        multidim_average: str = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        threshold_min: float = 0.0,
        threshold_max: float = 1.0,
        num_thresholds: int = 101,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_thresholds < 2:
            raise ValueError("num_thresholds must be >= 2.")
        if threshold_min >= threshold_max:
            raise ValueError("threshold_min must be smaller than threshold_max.")

        self.num_labels = num_labels
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.num_thresholds = num_thresholds
        self.best_threshold: float = 0.5

        self.add_state("preds", default=torch.zeros((0, num_labels), dtype=torch.float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros((0, num_labels), dtype=torch.int), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds = torch.cat([self.preds, preds.detach()])
        self.target = torch.cat([self.target, target.detach()])

    def _objective(self, score: Tensor) -> Tensor:
        if score.ndim == 0:
            return score
        finite = score[torch.isfinite(score)]
        if finite.numel() == 0:
            return torch.tensor(float("-inf"), device=score.device, dtype=score.dtype)
        return finite.mean()

    def compute(self) -> Tensor:
        if self.preds.numel() == 0:
            return torch.tensor(0.0, device=self.preds.device)

        preds = self.preds
        target = self.target

        if preds.is_floating_point() and ((preds < 0).any() or (preds > 1).any()):
            preds = preds.sigmoid()

        thresholds = torch.linspace(
            self.threshold_min,
            self.threshold_max,
            self.num_thresholds,
            device=preds.device,
            dtype=preds.dtype,
        )

        best_score: Tensor | None = None
        best_obj: Tensor | None = None
        best_thr = self.best_threshold

        for thr in thresholds:
            score = multilabel_f1_score(
                preds,
                target,
                num_labels=self.num_labels,
                threshold=float(thr.item()),
                average=self.average,
                multidim_average=self.multidim_average,
                ignore_index=self.ignore_index,
                validate_args=self.validate_args,
                zero_division=self.zero_division,
            )
            obj = self._objective(score)
            if best_obj is None or obj > best_obj:
                best_obj = obj
                best_score = score
                best_thr = float(thr.item())

        self.best_threshold = best_thr
        assert best_score is not None
        return best_score
