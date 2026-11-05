import torch
import torch.nn as nn


def supported_optimizers():
    return ("sgd", "adam", "adamw")


def create_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    assert isinstance(model, nn.Module)
    assert name in supported_optimizers(), f"Unsupported optimizer: {name}. Must be one of {supported_optimizers()}"

    # param_groups = model.parameters()
    param_groups = filter(lambda p: p.requires_grad, model.parameters())  # save memory

    if name == "sgd":
        # poor for training, unrecommended
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-7,
        )

    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-7,
        )

    return optimizer
