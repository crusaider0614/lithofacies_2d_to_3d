"""Module for injecting the auxiliary "info" channels into a feature map."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoChannelAppend2d(nn.Module):
    """Concatenate `info` to `x` along channels, resizing `info` bilinearly if the spatial sizes differ.

    Takes a tuple (x, info) so it can sit at the head of an nn.Sequential.
    """

    def __init__(self):
        super(InfoChannelAppend2d, self).__init__()

    def forward(self, input_tuple):
        x, info = input_tuple
        if x.shape[2:] != info.shape[2:]:
            info_reshape = F.interpolate(info, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
            res = torch.cat((x, info_reshape), dim=1)
        else:
            res = torch.cat((x, info), dim=1)
        return res
