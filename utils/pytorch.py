"""PyTorch helpers."""
import torch.nn as nn


def init_weights(module):
    """Xavier-uniform init for Linear/Conv layers (use with model.apply).

    Layers flagged with `_no_init = True` are skipped, so their custom init is kept
    (e.g. the zero-initialized output projection of NonLocalHorizontalBlock).
    """
    if hasattr(module, '_no_init') and module._no_init:
        return

    target_layer = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
    if isinstance(module, target_layer):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
