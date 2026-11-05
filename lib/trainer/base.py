"""Generic Framwork for Supervised Training Model"""
import warnings
warnings.filterwarnings("ignore", message=".*rich is experimental.*")

import os
import types
import torch
import inspect
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from lib.utils.entry import parse_from_cli
from ..metric import (
    MetricCollection,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelBestF1Score,
    MultilabelAveragePrecision,
    MultilabelAveragePrecision2,
)
from icecream import ic
from copy import deepcopy
from tqdm.rich import tqdm
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from typing import Literal, Annotated, OrderedDict
from pydantic import BaseModel, confloat, conint, field_validator

from .. import transforms as t
from ..optim import create_optimizer, create_scheduler, create_sequential_scheduler
from ..dataset import create_dataset, supported_multilabel_datasets, supported_multiclass_datasets
from ..loss import BceWithLogitsLoss
from ..utils import (
    AverageMeter,
    Checkpointer,
    GradScaler,
    Collector,
    autocast,
    best_gpu,
    init_seed,
    init_cuda_environment,
    is_redirected,
)

def create_metric(num_classes, type):
    if type == "multi-class":
        return MetricCollection({
            "mAP": MulticlassAveragePrecision(num_classes=num_classes),
            "accuracy": MulticlassAccuracy(num_classes=num_classes),
            "f1-score": MulticlassF1Score(num_classes=num_classes),
        })
    elif type == "multi-label":
        return MetricCollection({
            "mAP": MultilabelAveragePrecision(num_labels=num_classes),
            "mAP2": MultilabelAveragePrecision2(num_labels=num_classes),
            "accuracy": MultilabelAccuracy(num_labels=num_classes),
            "f1(best)": MultilabelBestF1Score(num_labels=num_classes),
        })
    else:
        raise ValueError(f"unsupported metric type ‘{type}’")


def create_datasets(cfg: DictConfig):
    name = cfg.dataset
    downsample_ratio = cfg.downsample_ratio
    valid_transform = t.Compose([t.Resize((cfg.image_size, cfg.image_size)), t.ToTensor(), t.Normalize()])

    train_dataset = create_dataset(
        name,
        divide="train",
        downsample_ratio=downsample_ratio,
        add_index=True,
        transform=valid_transform,
    )
    valid_dataset = create_dataset(
        name,
        divide="val",
        downsample_ratio=downsample_ratio,
        add_index=True,
        transform=valid_transform,
    )
    test_dataset = create_dataset(
        name,
        divide="test",
        downsample_ratio=downsample_ratio,
        add_index=True,
        transform=valid_transform,
    )

    return train_dataset, valid_dataset, test_dataset


def create_loader(dataset, cfg: DictConfig, *, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=len(dataset) < 10000,
    )


def create_loaders(train_dataset, valid_dataset, test_dataset, cfg: DictConfig):
    train_loader = create_loader(train_dataset, cfg, shuffle=True)
    valid_loader = create_loader(valid_dataset, cfg, shuffle=False)
    test_loader = create_loader(test_dataset, cfg, shuffle=False)
    return train_loader, valid_loader, test_loader


def create_optimizer(model, cfg: DictConfig):
    return create_optimizer(model, name=cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay)


def create_sequential_scheduler(optimizer, cfg: DictConfig):
    return create_sequential_scheduler(
        optimizer,
        name1=cfg.warmup_scheduler,
        gamma1=cfg.warmup_scheduler_gamma,
        name2=cfg.scheduler,
        gamma2=cfg.scheduler_gamma,
        warmup_epochs=cfg.warmup_epochs,
    )


def create_scheduler(optimizer, cfg: DictConfig):
    return create_scheduler(optimizer, name=cfg.scheduler, gamma=cfg.scheduler_gamma)


def init_environment(cfg: DictConfig):
    if cfg.device == "cuda":
        cfg.device = f"cuda:{best_gpu()}"

    if cfg.show_progress is None:
        cfg.show_progress = not is_redirected()
    else:
        cfg.show_progress = bool(cfg.show_progress)

    init_seed(cfg.seed)
    init_cuda_environment()

    # torch.autograd.set_detect_anomaly(True)  # suffer efficiency
    torch.multiprocessing.set_sharing_strategy("file_system")

    ic.configureOutput(includeContext=True)
    ic.enabled = cfg.debug


def learnable_state_dict(model: torch.nn.Module, *args, **kwargs) -> OrderedDict[str, torch.Tensor]:
    if hasattr(model, "learnable_parameters") is False:
        return model.state_dict(*args, **kwargs)
    learnable_names = [
        name for name, param in model.named_parameters() if any(param is p for p in model.learnable_parameters())
    ]
    return OrderedDict(
        {name: data for name, data in model.state_dict(*args, **kwargs).items() if name in learnable_names}
    )


class BaseConfig(BaseModel):
    model_config = {"extra": "forbid"}

    # General training settings
    amp: bool = True
    seed: int = 123
    show_progress: bool | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    debug: bool = True

    # Training settings
    max_epochs: Annotated[int, conint(ge=1)] = 50
    batch_size: Annotated[int, conint(ge=1)] = 64
    patience: Annotated[int, conint(ge=1)] = 3

    # Dataset and input settings
    dataset: str = "voc2007"
    image_size: Literal[224, 448] = 224
    downsample_ratio: Annotated[float, confloat(ge=0.0, le=1.0)] = 1.0

    # Storage settings
    save_summary: str | None = None

    @field_validator("device")
    def validate_device(cls, value):
        # ["cpu", "cuda", "cuda:0", "cuda:1", ..., "cuda:7"]
        valid_devices = ["cpu"]
        if torch.cuda.is_available():
            valid_devices += ["cuda"]
            valid_devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if value not in valid_devices:
            raise ValueError(f"Invalid device: {value}. Must be one of {valid_devices}.")
        return value

    @field_validator("dataset")
    def validate_dataset(cls, value):
        if value not in supported_multilabel_datasets():
            raise ValueError(f"Invalid dataset: {value}. Must be one of {supported_multilabel_datasets()}.")
        return value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}={repr(v)}" for k, v in OmegaConf.structured(self.model_dump()).items())
            + ")"
        )

    def __str__(self):
        return self.__repr__()

    def highlighted_str(self, cli_config: DictConfig):
        NONE = "\033[0m"
        GREEN = "\033[0;32m"
        GRAY = "\033[90m" 
        DELETE = "\033[9m"
        ORANGE = "\033[38;5;214m"  # orange-ish for overridden fields

        # helper: compare two FieldInfo-like objects for meaningful differences
        def field_differs(f1, f2):
            # Compare key attributes; fall back to repr if necessary
            a1 = getattr(f1, "annotation", None)
            a2 = getattr(f2, "annotation", None)
            if a1 != a2:
                return True
            d1 = getattr(f1, "default", ...) 
            d2 = getattr(f2, "default", ...)
            if d1 != d2:
                return True
            # final fallback: string compare of objects
            return repr(f1) != repr(f2)

        # return set of field names defined in cls but not inherited from bases
        # if include_overrides=True, include names that exist in bases but differ (overrides)
        def get_own_model_fields(cls, include_overrides: bool = False):
            cls_keys = set(cls.model_fields.keys())
            base_keys = set()
            for base in cls.__mro__[1:]:
                mf = getattr(base, "model_fields", None)
                if mf:
                    base_keys |= set(mf.keys())

            if not include_overrides:
                return cls_keys - base_keys

            # include overrides: include fields that are in bases but differ
            overrides = set()
            for k in cls_keys & base_keys:
                for base in cls.__mro__[1:]:
                    mf = getattr(base, "model_fields", None)
                    if mf and k in mf:
                        if field_differs(mf[k], cls.model_fields[k]):
                            overrides.add(k)
                            break
            return (cls_keys - base_keys) | overrides

        # choose behavior: set include_overrides True to treat overrides as "own"
        own_fields = get_own_model_fields(type(self), include_overrides=True)

        # also compute explicit overrides set for separate marking if desired
        override_fields = get_own_model_fields(type(self), include_overrides=True) - get_own_model_fields(type(self), include_overrides=False)

        # print(f"{own_fields = }")
        # print(f"{override_fields = }")

        def hightlight(k, v):
            # hide explicit None-annotation fields
            ann = type(self).model_fields.get(k).annotation
            # CLI-provided -> green
            if k in cli_config:
                return f"{GREEN}{k}={repr(v)}{NONE}"
            if ann is type(None):
                return None
            # inherited field -> gray
            if k not in override_fields and k not in own_fields:
                return f"{GRAY}{k}={repr(v)}{NONE}"
            return f"{k}={repr(v)}"

        return (
            f"{self.__class__.__name__}("
            + ", ".join([
                hightlight(k, v)
                for k, v in OmegaConf.structured(self.model_dump()).items()
                if hightlight(k, v) is not None
            ])
            + ")"
        )


class BaseTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.amp = cfg.amp
        self.device = cfg.device
        self.max_epochs = cfg.max_epochs

        self.train_dataset, self.valid_dataset, self.test_dataset = self.create_datasets(cfg)
        self.train_loader, self.valid_loader, self.test_loader = self.create_loaders(
            self.train_dataset, self.valid_dataset, self.test_dataset, cfg
        )
        self.model = self.create_model(self.train_dataset, cfg).to(self.device)
        self.optimizer = self.create_optimizer(self.model, cfg)
        self.scheduler = self.create_scheduler(self.optimizer, cfg)
        self.scaler = GradScaler(device=self.device.split(":")[0])
        self.checkpointer = Checkpointer(patience=cfg.patience, save_fn=self.save_fn, delta=-0.01)

        self.criterion = self.create_criterion(cfg).to(self.device)
        self.metric = self.create_metric(self.train_dataset.num_classes, cfg).to(self.device)

        self.collector = Collector()
        self.summary = {
            "name": self.__class__.__name__,
            "cfg": self.cfg.model_dump(),
            "time": str(datetime.now()),
            "state_dict": learnable_state_dict(self.model), # self.model.state_dict(),
            "train_info": [],
            "test_info": [],
            "code": inspect.getsource(self.__class__),
        }

        # for amp stablility
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    @staticmethod
    def init_environment(cfg: DictConfig):
        init_environment(cfg)

    def train(self):
        for self.epoch in range(self.max_epochs):
            if self.checkpointer.early_stop():
                break

            loss_summary: dict = self.train_one_epoch()
            eval_summary: dict = self.eval_one_epoch(mode="valid")
            info_summary: dict = {
                "epoch": self.epoch,
                # "lr": self.scheduler.get_last_lr()[0]
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            print(
                f"epoch: {info_summary['epoch']:d}",
                f"lr: {info_summary['lr']:.1e}",
                *[f"{k}: {v:.4f}" for k, v in loss_summary.items()],
                *[f"{k}: {v:.2%}" for k, v in eval_summary.items()],
                sep=", ",
            )

            self.summary["train_info"].append({**info_summary, **loss_summary, **eval_summary})
            self.checkpointer.update(score=self.best_score(eval_summary))
            self.scheduler.step()

        ckpt = self.checkpointer.best_checkpoint
        if ckpt is not None:
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"load model from epoch {ckpt['epoch']:d} with highest score {ckpt['score']:.2%}")

    def test(self):
        eval_summary = self.eval_one_epoch(mode="test")

        print(
            "test done",
            *[f"{k}: {v:.2%}" for k, v in eval_summary.items()],
            sep=", ",
        )

        self.summary["test_result"] = self.collector.compute()
        self.summary["test_info"].append(eval_summary)


    def train_one_epoch(self) -> dict:
        loss_summary = AverageMeter()
        self.model.train()

        for batch in self.tqdm(self.train_loader, desc="train"):
            batch_summary: dict = self.forward_backward(batch)
            loss_summary.update(**batch_summary)

        return loss_summary.average_value()

    def eval_one_epoch(self, mode: Literal["valid", "test"]) -> dict:
        loader = {"valid": self.valid_loader, "test": self.test_loader}[mode]
        self.model.eval()

        self.metric.reset()
        self.collector.reset()
        for batch in self.tqdm(loader, desc=mode):
            batch_summary: dict = self.forward(batch, mode=mode)
            self.collector.update(**batch_summary)
            self.metric.update(**batch_summary)

        return {k: v.cpu().item() for k, v in self.metric.compute().items()}

    def forward_backward(self, batch) -> dict:
        image, label, indices = batch
        image, label = image.to(self.device), label.to(self.device)

        with self.autocast():
            logits = self.model(image)
            loss = self.criterion(logits.float(), label.long())

        self.step(loss)

        loss_summary = {"loss": loss.cpu().item()}
        return loss_summary

    @torch.inference_mode()
    def forward(self, batch, mode: Literal["valid", "test"]) -> dict:
        image, label, indices = batch
        image, label = image.to(self.device), label.to(self.device)

        with self.autocast():
            logits = self.model(image)

        return {"indices": indices, "preds": logits.float(), "target": label.long()}

    def step(self, loss: Tensor):
        if self.amp:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def autocast(self):
        return autocast(device_type=self.device.split(":")[0], enabled=self.amp, dtype=torch.bfloat16)

    def tqdm(self, loader: DataLoader, desc: str = ""):
        return tqdm(
            loader,
            desc=desc,
            leave=False,
            disable=not self.cfg.show_progress,
            mininterval=1,
            # disable=False,
            # mininterval=0.1 if self.cfg.show_progress else 600,
        )

    def best_score(self, eval_summary: dict) -> float:
        return eval_summary["mAP"]

    def save_fn(self, score: float) -> dict:
        return deepcopy({
            "epoch": self.epoch,
            "score": score,
            # "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "state_dict": {k: v.cpu() for k, v in learnable_state_dict(self.model).items()},
        })

    def create_datasets(self, cfg: DictConfig):
        return create_datasets(cfg)

    def create_loaders(self, train_dataset, valid_dataset, test_dataset, cfg: DictConfig):
        return create_loaders(train_dataset, valid_dataset, test_dataset, cfg)

    def create_model(self, dataset, cfg: DictConfig) -> torch.nn.Module:
        raise NotImplemented

    def create_optimizer(self, model, cfg: DictConfig) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5, eps=1e-7)

    def create_scheduler(self, optimizer, cfg: DictConfig) -> LRScheduler:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

    def create_criterion(self, cfg: DictConfig) -> torch.nn.Module:
        return BceWithLogitsLoss(reduction="mean")

    def create_metric(self, num_classes, cfg: DictConfig) -> MetricCollection:
        return create_metric(num_classes, "multi-label")


class MulticlassBaseConfig(BaseConfig):
    dataset: str = "tinyimagenet"

    @field_validator("dataset")
    def validate_dataset(cls, value):
        if value not in supported_multiclass_datasets():
            raise ValueError(f"Invalid dataset: {value}. Must be one of {supported_multiclass_datasets()}.")
        return value


class MulticlassBaseTrainer(BaseTrainer):
    def create_criterion(self, cfg):
        return torch.nn.CrossEntropyLoss(reduction="mean")
    
    def create_metric(self, num_classes, cfg):
        return create_metric(num_classes, "multi-class")

    def forward_backward(self, batch) -> dict:
        image, label, idxs = batch
        image, label = image.to(self.device), label.argmax(dim=-1).to(self.device)

        with self.autocast():
            logits = self.model(image)
            loss = self.criterion(logits.float(), label.long())

        self.step(loss)

        loss_summary = {"loss": loss.cpu().item()}
        return loss_summary

    @torch.inference_mode()
    def forward(self, batch, mode: Literal["valid", "test"]) -> dict:
        image, label, idxs = batch
        image, label = image.to(self.device), label.argmax(dim=-1).to(self.device)

        with self.autocast():
            logits = self.model(image)

        return {"preds": logits.float(), "target": label.long()}


def entry(config, trainer, *, train: bool = True, test: bool = True):
    try:
        cli_cfg = parse_from_cli()
        cfg = config(**cli_cfg)
    except Exception as e:
        print(e)
        return

    trainer.init_environment(cfg)
    print(cfg.highlighted_str(cli_cfg))

    print("initializing trainer...")
    trainer = trainer(cfg)
    print(f"{trainer.train_dataset = }")
    print(f"{trainer.valid_dataset = }")
    print(f"{trainer.test_dataset = }")

    if train:
        print("start training...")
        trainer.train()
    if test:
        print("start testing...")
        trainer.test()
    print("done.")

    if cfg.save_summary is not None:
        torch.save(trainer.summary, cfg.save_summary)
        print(f"summary saved at {cfg.save_summary}")
