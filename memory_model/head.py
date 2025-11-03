import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalTrajectoryHead(nn.Module):
    def __init__(self, d_in: int, t_out: int, K: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.t_out, self.K = t_out, K
        self.in_norm = nn.LayerNorm(d_in)
        self.enc = nn.GRU(input_size=d_in, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, K * t_out * 2 + K)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        x = self.in_norm(x)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, (h_n) = self.enc(x)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        else:
            h, h_n = self.enc(x)
        h = h[:, -1]
        h = h_n.transpose(0, 1).reshape(x.size(0), -1)  # [B, 2H]
        y = self.proj(h)
        B = y.size(0)
        traj = y[:, : self.K * self.t_out * 2].view(B, self.K, self.t_out, 2)
        logits = y[:, self.K * self.t_out * 2:].view(B, self.K)
        return traj, logits
