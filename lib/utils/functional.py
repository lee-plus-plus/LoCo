import torch
import random
import numpy as np
import os
import sys
import warnings
from contextlib import ContextDecorator


def get_model_device(model):
    return next(model.parameters()).device


class ignore_prints(ContextDecorator):
    def __init__(self, no_stdout=True, no_stderr=False):
        super().__init__()
        self.no_stdout = no_stdout
        self.no_stderr = no_stderr

    def __enter__(self):
        if self.no_stdout:
            self.prev_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.no_stderr:
            self.prev_stdout = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.no_stdout:
            sys.stdout.close()
            sys.stdout = self.prev_stdout
        if self.no_stderr:
            sys.stderr.close()
            sys.stderr = self.prev_stderr


class ignore_warnings(ContextDecorator):
    def __enter__(self):
        self.catcher = warnings.catch_warnings()
        self.catcher.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, exc_type, exc_value, traceback):
        self.catcher.__exit__(exc_type, exc_value, traceback)


class local_random(ContextDecorator):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def __enter__(self):
        self.prev_random_state = random.getstate()
        self.prev_torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.prev_cuda_state = torch.cuda.random.get_rng_state()
        self.prev_np_state = np.random.get_state()

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.prev_random_state)
        torch.random.set_rng_state(self.prev_torch_state)
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(self.prev_cuda_state)
        np.random.set_state(self.prev_np_state)
