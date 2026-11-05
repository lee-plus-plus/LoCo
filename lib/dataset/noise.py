import torch


def add_partial_noise(
    labels: torch.Tensor,
    noise_rate: float,
) -> torch.Tensor:
    """add partial noise into labels with uniform probability"""
    noise = torch.rand(labels.shape) < noise_rate
    partial_labels = (labels.long() | noise.long()).long()
    return partial_labels


def add_imbalanced_noise(
    labels: torch.Tensor,
    pos_noise_rate: float,
    neg_noise_rate: float,
) -> torch.Tensor:
    """add partial noise into labels with uniform probability"""
    pos_noise = torch.rand(labels.shape) < pos_noise_rate
    neg_noise = torch.rand(labels.shape) < neg_noise_rate
    noise = ((labels == 1) & pos_noise) | ((labels == 0) & neg_noise)
    noisy_labels = (labels.long() ^ noise.long()).long()
    return noisy_labels