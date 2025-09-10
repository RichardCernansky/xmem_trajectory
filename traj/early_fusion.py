import torch
import torch.nn as nn

class EarlyFusionAdapter(nn.Module):
    def __init__(self, n_lidar: int, mode: str = "concat", out_channels: int = 3, use_bn: bool = False):
        super().__init__()
        assert mode in ("concat", "sum")
        self.mode = mode
        in_ch = 3 + n_lidar if mode == "concat" else n_lidar
        self.proj = nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity() # different scales of RGB and LiDAR so need of batch norm
        with torch.no_grad():
            w = self.proj.weight
            w.zero_()
            if mode == "concat":
                for c in range(min(out_channels, 3)):
                    w[c, c, 0, 0] = 1.0

    def forward(self, rgb: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([rgb, lidar], dim=1)
            y = self.proj(x)
        else:
            y = rgb + self.proj(lidar)
        return self.bn(y)
