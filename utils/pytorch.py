import torch.nn as nn


def init_weights(module):
    if hasattr(module, '_no_init') and module._no_init:
        return

    target_layer = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
    if isinstance(module, target_layer):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
