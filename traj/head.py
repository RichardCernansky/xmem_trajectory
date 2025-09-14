# traj/head.py
import torch
import torch.nn as nn

class TrajectoryHead(nn.Module):
    """
    Input:  [B, T_feat, D]
    Output: [B, F, 2]  (offsets)
    """
    def __init__(self, d_in: int, d_hid: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_size=d_in, hidden_size=d_hid, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, 2 * horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T_feat,D]
        h, _ = self.gru(x)                 # [B,T_feat,d_hid]
        h_last = h[:, -1]                  # [B,d_hid]
        out = self.mlp(h_last)             # [B,2*horizon]
        return out.view(-1, self.horizon, 2)  # [B,F,2]
