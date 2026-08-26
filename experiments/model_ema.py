"""Minimal state-dict exponential moving average for independently scored checkpoints."""

import torch


class StateDictEMA:
    """Track an EMA of all floating state and copy non-floating buffers exactly."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be strictly between 0 and 1")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model, decay=None):
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError("model state keys changed after EMA initialization")
        applied_decay = self.decay if decay is None else float(decay)
        if not 0.0 < applied_decay < 1.0:
            raise ValueError("applied EMA decay must be strictly between 0 and 1")
        one_minus_decay = 1.0 - applied_decay
        for key, value in current.items():
            shadow = self.shadow[key]
            if torch.is_floating_point(shadow):
                shadow.mul_(applied_decay).add_(value.detach(), alpha=one_minus_decay)
            else:
                shadow.copy_(value.detach())

    def state_dict(self):
        return self.shadow
