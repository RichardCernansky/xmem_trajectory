import torch
from torch import nn

class EarlyFusionAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(5, 3, kernel_size=1, bias=True)
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            w = torch.zeros(3, 5, 1, 1)
            w[:3, :3, 0, 0] = torch.eye(3)
            self.conv.weight.copy_(w)
            self.conv.bias.zero_()

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        y = self.conv(x)
        return y.view(b, t, 3, h, w)
