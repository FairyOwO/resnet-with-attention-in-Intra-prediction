import torch
from torch import nn, Tensor
from torch.nn import init, functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import *
from functools import partial
from itertools import permutations


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 4, actv_builder=nn.ReLU) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            actv_builder(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False),
        )
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                init.constant_(mod.bias, 0)

    def forward(self, x: Tensor) -> Tensor:

        avg_out = self.shared_mlp(self.avg(x))
        max_out = self.shared_mlp(self.max(x))
        weight = (avg_out + max_out).sigmoid()
        out = x * weight.expand_as(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels, *,
                 norm_builder=nn.Identity,
                 actv_builder=nn.ReLU,
                 pre_actv=False,
                 bias=True,
                 ):
        super().__init__()
        self.actv = actv_builder()
        self.pre_actv = pre_actv
        if self.pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
                norm_builder(),
                actv_builder(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            )

        else:
            self.res_unit = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
                norm_builder(),
                actv_builder(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=bias),
                norm_builder(),
            )
        self.ca = ChannelAttention(channels, actv_builder=actv_builder)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            conv_channels: int,
            num_blocks: int,
            n: int,
            *,
            actv_builder: nn.Module = nn.ReLU,
            norm_builder: nn.Module = nn.Identity,
            bias: bool = True,
            pre_actv: bool = False,
    ):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResBlock(
                    channels=conv_channels,
                    actv_builder=actv_builder,
                    norm_builder=norm_builder,
                    bias=bias,
                    pre_actv=pre_actv,
                )
            )
        layers = [nn.Conv2d(in_channels, conv_channels, 3, 1, 1, bias=bias), ]
        if pre_actv:
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            layers += [norm_builder(), actv_builder(), *blocks]
        layers += [nn.Conv2d(conv_channels, 32, 3, 1,1),
                   actv_builder(),
                   nn.Conv2d(32, in_channels, (1, 2*n+1), 1),]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return self.net(x)
