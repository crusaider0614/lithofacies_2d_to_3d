import torch.nn as nn


def init_weights(module):
    if hasattr(module, '_no_init') and module._no_init:
        return

    target_layer = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
    if isinstance(module, target_layer):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def get_n_params(model):
    n_whole_param = 0
    for parameter in list(model.parameters()):
        n_param = 1
        for len_dim in list(parameter.size()):
            n_param = len_dim * n_param
        n_whole_param += n_param
    return n_whole_param


def load_network_state_dict(network, state_dict, strict=True):
    network_dict = network.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in network_dict}
    network_dict.update(state_dict)
    network.load_state_dict(network_dict, strict=strict)
