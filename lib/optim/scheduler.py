from torch.optim.lr_scheduler import SequentialLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
from ..utils import ignore_warnings


def supported_scheduler():
    return ("exponential", "cosine", "constant")


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, lr: float):
        self.lr = lr
        super().__init__(optimizer)

    def get_lr(self):
        return [self.lr for base_lr in self.base_lrs]


class SilentSequentialLR(SequentialLR):
    @ignore_warnings()
    def step(self):
        super().step()


def create_scheduler(optimizer, name, gamma):
    assert name in supported_scheduler(), f"Unsupported scheduler: {name}. Must be one of {supported_scheduler()}"

    if name == "exponential":
        return ExponentialLR(optimizer, gamma=gamma)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=gamma)
    elif name == "constant":
        return ConstantLR(optimizer, lr=gamma)


def create_sequential_scheduler(optimizer, name1, gamma1, name2, gamma2, warmup_epochs):
    scheduler1 = create_scheduler(optimizer, name1, gamma1)
    scheduler2 = create_scheduler(optimizer, name2, gamma2)
    scheduler = SilentSequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_epochs])
    return scheduler
