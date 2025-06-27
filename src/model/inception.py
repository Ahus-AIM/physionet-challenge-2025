from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv1d:
    """Convolution with padding"""
    conv_layer = nn.Conv1d(
        in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False
    )
    return conv_layer


class InceptionBlock1d(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        kernel_sizes: List[int],
        bottleneck_channels: int,
        use_bottleneck: bool,
    ) -> None:
        super(InceptionBlock1d, self).__init__()

        self.bottleneck: nn.Module
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(num_in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.bottleneck = nn.Identity()

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels if use_bottleneck else num_in_channels,
                    num_out_channels,
                    kernel_size=k,
                    padding=(k - 1) // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(num_in_channels, num_out_channels, kernel_size=1, stride=1, bias=False),
        )

        self.batch_norm = nn.BatchNorm1d(num_out_channels * (len(kernel_sizes) + 1))
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        x_convs = [conv(x_bottleneck) for conv in self.convs]
        x_pool = self.pool_conv(x)
        x_out = torch.cat(x_convs + [x_pool], dim=1)
        x_out = self.batch_norm(x_out)
        x = self.gelu(x_out)
        return x


class Shortcut1d(nn.Module):
    def __init__(self, num_in_channels: int, num_out_channels: int) -> None:
        super(Shortcut1d, self).__init__()
        self.conv = nn.Conv1d(num_in_channels, num_out_channels, kernel_size=1, stride=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(num_out_channels)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x_shortcut = self.conv(x)
        x_shortcut = self.batch_norm(x_shortcut)
        x = nn.GELU()(x_shortcut + residual)
        return x


class InceptionNetwork(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_in_channels: int,
        num_out_channels: int,
        kernel_sizes: List[int],
        use_bottleneck: bool,
        bottleneck_channels: int,
        num_classes: int,
        residual: bool,
    ) -> None:
        super(InceptionNetwork, self).__init__()

        self.blocks = nn.ModuleList()
        self.residual = residual

        for i in range(num_blocks):
            self.blocks.append(
                InceptionBlock1d(
                    num_in_channels=num_in_channels if i == 0 else num_out_channels * 4,
                    num_out_channels=num_out_channels,
                    bottleneck_channels=bottleneck_channels,
                    kernel_sizes=kernel_sizes,
                    use_bottleneck=use_bottleneck,
                )
            )

        if residual:
            self.shortcuts = nn.ModuleList(
                [
                    Shortcut1d(num_in_channels if i == 0 else num_out_channels * 4, num_out_channels * 4)
                    for i in range(num_blocks // 3)
                ]
            )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_out_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_res = x

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.residual and (i % 3 == 2):
                x = self.shortcuts[i // 3](input_res, x)
                input_res = x.clone()

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionNetworkWithDownsampling(nn.Module):
    def __init__(
        self,
        num_blocks: int = 6,
        num_in_channels: int = 8,
        num_out_channels: int = 32,
        kernel_sizes: List[int] = [9, 19, 39],
        use_bottleneck: bool = True,
        bottleneck_channels: int = 32,
        num_classes: int = 1,
        residual: bool = True,
    ) -> None:
        super(InceptionNetworkWithDownsampling, self).__init__()
        self.inception_network = InceptionNetwork(
            num_blocks=num_blocks,
            num_in_channels=num_out_channels,
            num_out_channels=num_out_channels,
            kernel_sizes=kernel_sizes,
            use_bottleneck=use_bottleneck,
            bottleneck_channels=bottleneck_channels,
            num_classes=num_classes,
            residual=residual,
        )
        self.down_layers = nn.Sequential(
            nn.Conv1d(num_in_channels, num_out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(num_out_channels),
            nn.GELU(),
            nn.Conv1d(num_out_channels, num_out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(num_out_channels),
            nn.GELU(),
        )
        self.num_in_channels = num_in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_layers(x)
        x = self.inception_network(x)
        return x

    def load_weights(self, path: str) -> None:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, OrderedDict):
            self.load_state_dict(loaded)
        elif isinstance(loaded, tuple):
            self.load_state_dict(loaded[0])
        else:
            print("Could not load weights")
