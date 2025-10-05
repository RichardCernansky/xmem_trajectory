import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_depth_channels(depths: torch.Tensor, dmin: float = 1.0, dmax: float = 80.0):
    """
    depths: (B,T,1,H,W) in meters; 0.0 => invalid
    returns:
      inv  : (B,T,1,H,W) inverse-depth in [0,1], masked by validity
      valid: (B,T,1,H,W) {0,1}
    """
    valid = (depths > 0).float()
    d = depths.clamp(min=dmin, max=dmax)
    inv = (1.0 / d - 1.0 / dmax) / (1.0 / dmin - 1.0 / dmax)  # [0,1], larger = nearer
    inv = inv * valid
    return inv, valid

class DepthEncoder(nn.Module):
    """
    Very small encoder for (inv_depth, valid) -> depth features.
    Works at the feature-map resolution (we downsample inputs to h×w).
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x_btchw: torch.Tensor) -> torch.Tensor:
        # x_btchw: (B*T, in_ch, h, w)
        return self.net(x_btchw)  # (B*T, out_ch, h, w)

class LateConcatFuser(nn.Module):
    """
    Concatenate XMem features and depth features; reduce back to XMem channel count.
    """
    def __init__(self, feat_c: int, depth_feat_c: int = 64):
        super().__init__()
        self.reduce = nn.Conv2d(feat_c + depth_feat_c, feat_c, kernel_size=1)

    def forward(self, feats: torch.Tensor, dfeat: torch.Tensor) -> torch.Tensor:
        """
        feats: (B,T,C,h,w)
        dfeat: (B,T,Cd,h,w)
        -> (B,T,C,h,w)
        """
        B, T, C, h, w = feats.shape
        x = torch.cat([feats.flatten(0,1), dfeat.flatten(0,1)], dim=1)  # (B*T, C+Cd, h, w)
        x = self.reduce(x).view(B, T, C, h, w)
        return x

def build_depth_features(depths: torch.Tensor, encoder: DepthEncoder, target_hw, dmin=1.0, dmax=80.0):
    """
    depths: (B,T,1,H,W) -> inv+valid -> resize to (h,w) -> encode -> (B,T,Cd,h,w)
    """
    B, T, _, H, W = depths.shape
    h, w = target_hw
    inv, valid = prepare_depth_channels(depths, dmin=dmin, dmax=dmax)          # (B,T,1,H,W) each
    x = torch.cat([inv, valid], dim=2)                                          # (B,T,2,H,W)
    x = F.interpolate(x.flatten(0,1), size=(h, w), mode='nearest')              # (B*T,2,h,w)
    dfeat = encoder(x).view(B, T, -1, h, w)                                     # (B,T,Cd,h,w)
    return dfeat
