"""Exponential Moving Average (EMA) for model weights during training."""
import copy
import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters.
    
    EMA improves generalization by smoothing parameter updates across training steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters with current model parameters."""
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if k in msd:
                if v.dtype.is_floating_point:
                    v.copy_(d * v + (1.0 - d) * msd[k].to(v.device, dtype=v.dtype))
                else:
                    v.copy_(msd[k].to(v.device))

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)
