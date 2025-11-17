import torch
import torch.nn as nn
import torch.nn.functional as F

class PPToFrames(nn.Module):
    def __init__(self, c_in: int, c_hidden: int = 64):
        super().__init__()
        self.norm = nn.GroupNorm(8, c_in)
        self.proj1 = nn.Conv2d(c_in, c_hidden, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.proj2 = nn.Conv2d(c_hidden, 3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = torch.sigmoid(x)
        return x
