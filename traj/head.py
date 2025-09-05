import torch, torch.nn as nn

class TrajectoryHead(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 256, horizon: int = 30):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hid, num_layers=1, batch_first=True)
        self.proj = nn.Linear(d_hid, 2)  # (dx, dy) per step
        self.horizon = horizon

    def forward(self, x_seq):  # [B, T_in, d_in]
        h, _ = self.gru(x_seq)                 # [B, T_in, d_hid]
        last = h[:, -1]                        # [B, d_hid]
        # Predict future offsets step-by-step from last hidden
        outs = []
        state = last.unsqueeze(1)              # [B,1,d_hid]
        for _ in range(self.horizon):
            dxdy = self.proj(state.squeeze(1)) # [B,2]
            outs.append(dxdy.unsqueeze(1))
            # simple autoregressive state update (optional: another GRUCell)
            state = state                      # keep state (minimal baseline)
        return torch.cat(outs, dim=1)          # [B, horizon, 2]
