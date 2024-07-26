import math

import torch
from torch.optim.lr_scheduler import LRScheduler


def cycle(dl):
    while True:
        for dt in dl:
            yield dt


def clean_text(text):
    cleaned_text = text.replace("\n", "")
    cleaned_text = cleaned_text.replace(" ", " ")
    cleaned_text = cleaned_text.replace("“", "\"")
    cleaned_text = cleaned_text.replace("”", "\"")
    cleaned_text = cleaned_text.replace("″", "\"")
    cleaned_text = cleaned_text.replace("’", "'")
    cleaned_text = cleaned_text.replace("‘", "'")
    cleaned_text = cleaned_text.replace("′", "'")

    return cleaned_text


def pad_sequence(array, max_length, pad_value=0):
    for subarray in array:
        while len(subarray) < max_length:
            subarray.append(pad_value)
    return array


class CosineAnnealingWithWarmRestartsLR(LRScheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int = 128,
                 cycle_steps: int = 512,
                 min_lr: float = 0.,
                 max_lr: float = 1e-3
                 ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.steps_counter = 0

        super().__init__(optimizer)

    def step(self, epoch=None):
        self.steps_counter += 1

        current_cycle_steps = self.steps_counter % self.cycle_steps

        if current_cycle_steps < self.warmup_steps:
            current_lr = self.min_lr + (self.max_lr - self.min_lr) * current_cycle_steps / self.warmup_steps
        else:
            current_lr = self.min_lr + (self.max_lr - self.min_lr) * \
                         (1 + math.cos(math.pi * (current_cycle_steps - self.warmup_steps) /
                                       (self.cycle_steps - self.warmup_steps))) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
