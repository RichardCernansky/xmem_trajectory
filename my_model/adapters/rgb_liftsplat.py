# models/rgb_bev/rgb_liftsplat.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Small 2D encoder (stride 8)
# -------------------------------
class _ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=2, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):  # (N,C,H,W) -> (N,cout,H/stride,W/stride)
        return self.net(x)

class _ImageEncoder2D(nn.Module):
    def __init__(self, out_ch=128):
        super().__init__()
        self.stem = _ConvBNReLU(3, 32, 7, 2, 3)  # /2
        self.b1   = _ConvBNReLU(32, 64, 3, 2, 1) # /4
        self.b2   = _ConvBNReLU(64, 128, 3, 2, 1)# /8
        self.proj = nn.Conv2d(128, out_ch, 1)
    def forward(self, x):  # (N,3,H,W) -> (N,out_ch,H/8,W/8)
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.proj(x)
        return x

# -------------------------------
# Intrinsics scaling to feature res
# -------------------------------
def _scale_K(K_scaled, H, W, Hp, Wp):
    """
    K_scaled: (...,3,3) at (H,W). Return intrinsics for (Hp, Wp).
    """
    Sy, Sx = float(Hp)/float(H), float(Wp)/float(W)
    Kp = K_scaled.clone()
    Kp[..., 0, 0] *= Sx; Kp[..., 1, 1] *= Sy
    Kp[..., 0, 2] *= Sx; Kp[..., 1, 2] *= Sy
    return Kp

# -------------------------------
# Lift & Splat (flatten BTC -> N)
# -------------------------------
class _LiftSplat(nn.Module):
    def __init__(self, H_bev, W_bev, x_min, x_max, y_min, y_max):
        super().__init__()
        self.Hb, self.Wb = int(H_bev), int(W_bev)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)

    def forward(self, feats, depths, K_scaled, T_cam_from_ego, H, W):
        """
        feats:   (B,T,C,Cr,Hp,Wp)     feature maps @ feature res
        depths:  (B,T,C,H,W)          LiDAR z-buffer @ image res (meters; 0=no hit)
        K_scaled:(B,T,C,3,3)          intrinsics @ image res (H,W)
        T_cam_from_ego:(B,T,C,4,4)    extrinsics (sensor<-ego)
        H,W:     original image size that K_scaled corresponds to
        Returns:
          BEV features averaged across cameras: (B,T,Cr,Hb,Wb)
        """
        device = feats.device
        dtype  = feats.dtype

        B, T, C, Cr, Hp, Wp = feats.shape
        N = B * T * C
        Hb, Wb = self.Hb, self.Wb
        rx = (self.x_max - self.x_min) / float(Wb)
        ry = (self.y_max - self.y_min) / float(Hb)

        # ---- 1) Resize depths to (Hp,Wp) in 4D, then reshape back
        depths = depths.to(dtype)                        # (B,T,C,H,W)
        d4 = depths.reshape(N, 1, H, W)                  # (N,1,H,W)
        d4 = F.interpolate(d4, size=(Hp, Wp), mode="nearest")  # (N,1,Hp,Wp)

        # ---- 2) Scale intrinsics to feature res and flatten
        Kp = _scale_K(K_scaled, H, W, Hp, Wp).to(dtype)  # (B,T,C,3,3)
        Kp = Kp.reshape(N, 3, 3)                         # (N,3,3)

        # ---- 3) Ego<-cam and flatten
        T_egofromcam = torch.linalg.inv(T_cam_from_ego).to(dtype)  # (B,T,C,4,4)
        T_egofromcam = T_egofromcam.reshape(N, 4, 4)               # (N,4,4)

        # ---- 4) Pixel grid (Hp,Wp) and broadcast to N
        yy = torch.arange(Hp, device=device, dtype=dtype)
        xx = torch.arange(Wp, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(yy, xx, indexing="ij")   # (Hp,Wp)
        uu = uu.unsqueeze(0).unsqueeze(0).expand(N, 1, Hp, Wp)  # (N,1,Hp,Wp)
        vv = vv.unsqueeze(0).unsqueeze(0).expand(N, 1, Hp, Wp)

        # ---- 5) Back-project to camera frame
        fx = Kp[:, 0, 0].view(N,1,1,1).expand(N,1,Hp,Wp)
        fy = Kp[:, 1, 1].view(N,1,1,1).expand(N,1,Hp,Wp)
        cx = Kp[:, 0, 2].view(N,1,1,1).expand(N,1,Hp,Wp)
        cy = Kp[:, 1, 2].view(N,1,1,1).expand(N,1,Hp,Wp)

        x_c = (uu - cx) / fx * d4                         # (N,1,Hp,Wp)
        y_c = (vv - cy) / fy * d4
        z_c = d4
        ones = torch.ones_like(z_c)
        Xc = torch.cat([x_c, y_c, z_c, ones], dim=1)      # (N,4,Hp,Wp)
        Xc = Xc.reshape(N, 4, Hp*Wp)                      # (N,4,HpWp)

        # ---- 6) Camera->ego
        T34 = T_egofromcam[:, :3, :4]                     # (N,3,4)
        Xe = torch.matmul(T34, Xc)                         # (N,3,HpWp)
        Xe = Xe.reshape(B, T, C, 3, Hp, Wp)               # (B,T,C,3,Hp,Wp)

        x_ego = Xe[:, :, :, 0]
        y_ego = Xe[:, :, :, 1]

        # Depth > 0 at feature res
        depths_rs = d4.view(B, T, C, Hp, Wp)              # (B,T,C,Hp,Wp)

        # ---- 7) Validity & BEV indices
        valid = (depths_rs > 0) & \
                (x_ego >= self.x_min) & (x_ego < self.x_max) & \
                (y_ego >= self.y_min) & (y_ego < self.y_max)

        ix = ((x_ego - self.x_min) / rx).floor().clamp(0, Wb-1).long()
        iy = ((y_ego - self.y_min) / ry).floor().clamp(0, Hb-1).long()
        lin = iy * Wb + ix                                  # (B,T,C,Hp,Wp)

        # ---- 8) Splat features into BEV and average across cameras
        feats_flat = feats.reshape(N, Cr, Hp*Wp)            # (N,Cr,HpWp)
        lin_flat   = lin.reshape(N, Hp*Wp)                  # (N,HpWp)
        valid_flat = valid.reshape(N, Hp*Wp)                # (N,HpWp)

        out = feats.new_zeros((N, Cr, Hb*Wb))
        cnt = feats.new_zeros((N, 1,  Hb*Wb))

        idx = lin_flat.masked_fill(~valid_flat, 0)          # (N,HpWp)
        # scatter_add along last dim
        out.scatter_add_(2, idx.unsqueeze(1).expand(-1, Cr, -1),
                         feats_flat * valid_flat.unsqueeze(1))
        cnt.scatter_add_(2, idx.unsqueeze(1),
                         torch.ones_like(feats_flat[:, :1]) * valid_flat.unsqueeze(1))

        out = out.view(B, T, C, Cr, Hb, Wb)
        cnt = cnt.view(B, T, C, 1,  Hb, Wb).clamp_min(1.0)
        bev = (out.sum(dim=2) / cnt.sum(dim=2))             # (B,T,Cr,Hb,Wb)
        return bev

# -------------------------------
# Top-level RGB Lift/Splat encoder
# -------------------------------
class RGBLiftSplatEncoder(nn.Module):
    def __init__(self, H_bev, W_bev, x_min, x_max, y_min, y_max, C_r=128):
        super().__init__()
        self.enc  = _ImageEncoder2D(out_ch=C_r)
        self.view = _LiftSplat(H_bev, W_bev, x_min, x_max, y_min, y_max)

    def forward(self, cam_imgs, cam_depths, cam_K_scaled, cam_T_cam_from_ego, H, W):
        """
        cam_imgs:           (B,T,C,3,H,W_img)
        cam_depths:         (B,T,C,H,W_img)
        cam_K_scaled:       (B,T,C,3,3)  intrinsics @ (H,W_img)
        cam_T_cam_from_ego: (B,T,C,4,4)
        H,W: ints of image size that K_scaled is defined for
        Returns: F_bev (B,T,C_r,H_bev,W_bev)
        """
        B, T, C, Ch, Himg, Wimg = cam_imgs.shape
        x = cam_imgs.reshape(B*T*C, Ch, Himg, Wimg)         # (BTC,3,H,W)
        f = self.enc(x)                                     # (BTC,C_r,Hp,Wp)
        Cr, Hp, Wp = f.shape[1], f.shape[2], f.shape[3]
        f = f.reshape(B, T, C, Cr, Hp, Wp)                  # (B,T,C,Cr,Hp,Wp)
        F_bev = self.view(f, cam_depths, cam_K_scaled, cam_T_cam_from_ego, H, W)
        return F_bev
