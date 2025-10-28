# models/adapters/bev_frames.py
import torch, torch.nn as nn

class _PosEnc2D(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        y = torch.linspace(-1,1,H).view(H,1).expand(H,W)
        x = torch.linspace(-1,1,W).view(1,W).expand(H,W)
        pe = torch.stack([y,x], dim=0)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, B, T):
        return self.pe.unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1,-1)  # (B,T,2,H,W)

class CamBEVToFrames(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.to3 = nn.Conv3d(c_in, 3, 1)
    def forward(self, F_cam):               # (B,T,C_in,H,W)
        x = F_cam.permute(0,2,1,3,4)
        return self.to3(x).permute(0,2,1,3,4)  # (B,T,3,H,W)

class LiDARToFrames(nn.Module):
    def __init__(self, H_bev, W_bev, with_posenc=True):
        super().__init__()
        self.pe = _PosEnc2D(H_bev, W_bev) if with_posenc else None
        in_ch = 4 + (2 if with_posenc else 0)
        self.to3 = nn.Conv3d(in_ch, 3, 1)
    def forward(self, lidar_bev_raw):       # (B,T,4,H,W)
        B,T,_,H,W = lidar_bev_raw.shape
        x = lidar_bev_raw
        if self.pe is not None:
            x = torch.cat([x, self.pe(B,T)], dim=2)    # (B,T,6,H,W)
        x = x.permute(0,2,1,3,4)
        return self.to3(x).permute(0,2,1,3,4)          # (B,T,3,H,W)
