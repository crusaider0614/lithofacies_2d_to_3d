import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from network.info_module import InfoPass, InfoChannelAppend2d
from torch.nn.utils.parametrizations import spectral_norm
from utils.pytorch import init_weights


class GroupNorm(nn.Module):
    def __init__(self, channels, groups=16):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class FPA(nn.Module):
    def __init__(self, in_channels, out_channels, is_norm=True):
        super(FPA, self).__init__()

        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
        )

        self.down2_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, stride=2, padding=2),
            GroupNorm(in_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )
        self.down2_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride=1, padding=2),
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )

        self.down3_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            GroupNorm(in_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )
        self.down3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x_glob = self.glob(x)
        x_glob = F.interpolate(x_glob, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        d2 = self.down2_1(x)
        d3 = self.down3_1(d2)

        d2 = self.down2_2(d2)
        d3 = self.down3_2(d3)

        d3 = F.interpolate(d3, size=(d2.size(2), d2.size(3)), mode="bilinear", align_corners=False)
        d2 = d2 + d3

        d2 = F.interpolate(d2, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = x * d2

        x = x + x_glob
        return x


class NonLocalHorizontalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(NonLocalHorizontalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, (kernel_size, 1), 1, (kernel_size // 2, 0))
        self.k = nn.Conv2d(channels, channels, (kernel_size, 1), 1, (kernel_size // 2, 0))
        self.v = nn.Conv2d(channels, channels, (kernel_size, 1), 1, (kernel_size // 2, 0))

        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out._no_init = True
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x):
        h_ = self.gn(x)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, nz, nx = q.shape
        ns = nx

        q = q.permute(0, 2, 1, 3).reshape(b * nz, c, ns)
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1, 3).reshape(b * nz, c, ns)
        v = v.permute(0, 2, 1, 3).reshape(b * nz, c, ns)

        attn = torch.bmm(q, k)
        attn = attn * (int(c) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, nz, c, nx).permute(0, 2, 1, 3)
        A = self.proj_out(A)

        return x + A


class UpSampleBlock(nn.Module):
    def __init__(self, channels, stride=2):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels, stride=2):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.AvgPool2d(kernel_size=stride, stride=stride),
        )

    def forward(self, x):
        return self.conv(x)


class InfoDecodeBlock(nn.Module):
    def __init__(self, ui_channels, xi_channels, out_channels, is_norm=True, is_attn=True, attn_kernel=3):
        super(InfoDecodeBlock, self).__init__()

        self.u_conv = UpSampleBlock(ui_channels)
        self.x_attn = NonLocalHorizontalBlock(xi_channels, kernel_size=attn_kernel) if is_attn else nn.Identity()
        self.network = nn.Sequential(
            nn.Conv2d(ui_channels + xi_channels, out_channels, 3, stride=1, padding=1),
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
            NonLocalHorizontalBlock(out_channels, kernel_size=attn_kernel) if is_attn else nn.Identity(),
        )

    def forward(self, u, x):
        u = self.u_conv(u)
        x = self.x_attn(x)

        if u.size(2) != x.size(2) or u.size(3) != x.size(3):
            u = F.interpolate(u, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        u = torch.cat([u, x], dim=1)
        out = self.network(u)
        return out


class InfoResBlock(nn.Module):
    def __init__(self, in_channels, info_channels, out_channels, stride=1, is_norm=True):
        super(InfoResBlock, self).__init__()

        self.stride = stride
        self.info_channels = info_channels

        self.pool = DownSampleBlock(in_channels, stride=stride) if stride != 1 else nn.Identity()

        self.bypass = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0) if (in_channels != out_channels) else nn.Identity()
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels + info_channels, out_channels, 3, stride=1, padding=1),
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

        self.last = nn.Sequential(
            GroupNorm(out_channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )

    def forward(self, input_tuple):
        x, info = input_tuple
        w = self.pool(x)

        bypass = self.bypass(w)

        if self.info_channels != 0:
            info_reshape = F.interpolate(info, size=(w.size(2), w.size(3)), mode="bilinear", align_corners=False)
            w = torch.cat((w, info_reshape), dim=1)
        residual = self.residual(w)
        out = self.last(bypass + residual)
        return out


class InfoUNet(nn.Module):
    def __init__(self, in_channels, info_channels, channels, out_channels, is_norm=True, is_attn=True):
        super(InfoUNet, self).__init__()

        self.conv = nn.Sequential(
            InfoChannelAppend2d(),
            nn.Conv2d(in_channels + info_channels, channels, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            GroupNorm(channels) if is_norm else nn.Identity(),
            nn.SiLU(inplace=True),
        )

        self.encode2 = nn.Sequential(
            InfoResBlock(1 * channels, info_channels, 1 * channels, is_norm=is_norm),
        )
        self.encode3 = nn.Sequential(
            InfoResBlock(1 * channels, 0, 2 * channels, stride=2, is_norm=is_norm),
        )
        self.encode4 = nn.Sequential(
            InfoResBlock(2 * channels, 0, 4 * channels, stride=2, is_norm=is_norm),
        )
        self.encode5 = nn.Sequential(
            InfoResBlock(4 * channels, 0, 8 * channels, stride=2, is_norm=is_norm),
        )

        self.center = nn.Sequential(
            FPA(8 * channels, 1 * channels, is_norm=is_norm),
            nn.AvgPool2d(2, 2),
        )

        self.decode5 = InfoDecodeBlock(1 * channels, 8 * channels, 1 * channels, is_norm=is_norm, is_attn=is_attn, attn_kernel=3)
        self.decode4 = InfoDecodeBlock(1 * channels, 4 * channels, 1 * channels, is_norm=is_norm, is_attn=is_attn, attn_kernel=3)
        self.decode3 = InfoDecodeBlock(1 * channels, 2 * channels, 1 * channels, is_norm=is_norm, is_attn=is_attn, attn_kernel=5)
        self.decode2 = InfoDecodeBlock(1 * channels, 1 * channels, 1 * channels, is_norm=is_norm, is_attn=is_attn, attn_kernel=5)

        self.logit = nn.Sequential(
            nn.Conv2d(4 * channels, out_channels, 1, stride=1, padding=0),
        )

    def forward(self, x, info):
        e1 = self.conv((x, info))
        e2 = self.encode2((e1, info))
        e3 = self.encode3((e2, None))
        e4 = self.encode4((e3, None))
        e5 = self.encode5((e4, None))

        f = self.center(e5)

        d5 = self.decode5(f,  e5)
        d4 = self.decode4(d5, e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)

        output_shape = (d2.size(2), d2.size(3))
        f = torch.cat(
            (
                d2,
                F.interpolate(d3, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(d4, size=output_shape, mode="bilinear", align_corners=False),
                F.interpolate(d5, size=output_shape, mode="bilinear", align_corners=False),
            ),
            dim=1,
        )

        logit = self.logit(f)
        return logit


def get_gen_model(cfg, additional_channel=0):
    model = InfoUNet(
        cfg.MODEL.A_CHANNELS + additional_channel,
        cfg.MODEL.INFO_CHANNELS,
        cfg.MODEL.GEN_CHANNELS,
        cfg.MODEL.B_CHANNELS,
        is_norm=cfg.MODEL.GEN_NORM,
    )

    # load the pre-trained model
    if "PRETRAINED" in cfg.MODEL.keys() and os.path.exists(cfg.MODEL.PRETRAINED) and os.path.isfile(cfg.MODEL.PRETRAINED):
        trained_model = torch.load(cfg.MODEL.PRETRAINED)
        trained_model = {k.replace("module.", ""): v for (k, v) in trained_model.items()}
        model.load_state_dict(trained_model, strict=True)
    return model


if __name__ == "__main__":
    device = "cuda:0"

    batch_size = 1
    gen = InfoUNet(1, 0, 32, 6, is_norm=True, is_attn=True)
    gen.apply(init_weights)
    gen = gen.to(device)

    x = torch.randn(batch_size, 1, 320, 320).to(device)
    print(x[:, 0: 0].shape)
    info = torch.randn(batch_size, 0, 320, 320).to(device)
    y = gen(x, x[:, 0: 0])
    # # y = y[:, :, :32, :32, :32]
    # # info = info[:, :, :32, :32, :32]
    print(y.shape)
    # print(y.shape)
