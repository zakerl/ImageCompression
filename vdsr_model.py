import torch
from torch import nn
from torch.nn import functional as F


class VDSR(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_layers: int, channels: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.intermediate_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

        self._initialize_weights()

    def forward(
        self, Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor, factor: int
    ) -> torch.Tensor:
        Y = F.interpolate(
            Y[:, None], scale_factor=factor, mode="bilinear", align_corners=True
        )
        U = F.interpolate(
            U[:, None], scale_factor=factor * 2, mode="bilinear", align_corners=True
        )
        V = F.interpolate(
            V[:, None], scale_factor=factor * 2, mode="bilinear", align_corners=True
        )
        x = torch.cat([Y, U, V], dim=1)
        residual = x
        x = self.conv1(x)
        x = self.intermediate_layers(x)
        out = self.conv2(x) + residual
        out = torch.clamp(out, 0, 1)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
