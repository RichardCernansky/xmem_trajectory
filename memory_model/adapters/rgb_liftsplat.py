# models/rgb_bev/rgb_liftsplat.py
import torch, torch.nn as nn, torch.nn.functional as F

class _ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=2, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class _ImageEncoder2D(nn.Module):
    def __init__(self, out_ch=128):
        super().__init__()
        self.stem = _ConvBNReLU(3, 32, 7, 2, 3)
        self.b1   = _ConvBNReLU(32, 64, 3, 2, 1)
        self.b2   = _ConvBNReLU(64, 128, 3, 2, 1)
        self.proj = nn.Conv2d(128, out_ch, 1)
    def forward(self, x):
        x = self.stem(x); x = self.b1(x); x = self.b2(x); x = self.proj(x)
        return x

def _scale_K(K_scaled, H, W, Hp, Wp):
    Sy, Sx = Hp/float(H), Wp/float(W)
    Kp = K_scaled.clone()
    Kp[...,0,0] *= Sx; Kp[...,1,1] *= Sy
    Kp[...,0,2] *= Sx; Kp[...,1,2] *= Sy
    return Kp

class _LiftSplat(nn.Module):
    def __init__(self, H_bev, W_bev, x_min, x_max, y_min, y_max):
        super().__init__()
        self.Hb, self.Wb = H_bev, W_bev
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
    def forward(self, feats, depths, K_scaled, T_cam_from_ego, H, W):
        B,T,C,Cr,Hp,Wp = feats.shape
        rx = (self.x_max - self.x_min)/float(self.Wb)
        ry = (self.y_max - self.y_min)/float(self.Hb)

        depths = F.interpolate(depths.unsqueeze(2).float(), size=(Hp,Wp), mode="nearest")[:,:, :,0]
        Kp = _scale_K(K_scaled, H, W, Hp, Wp)
        T_egofromcam = torch.linalg.inv(T_cam_from_ego)

        u = torch.arange(Wp, device=feats.device).float()
        v = torch.arange(Hp, device=feats.device).float()
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        uu = uu.view(1,1,1,1,Hp,Wp).expand(B,T,C,1,Hp,Wp)
        vv = vv.view(1,1,1,1,Hp,Wp).expand(B,T,C,1,Hp,Wp)

        fx = Kp[...,0,0].unsqueeze(-1).unsqueeze(-1)
        fy = Kp[...,1,1].unsqueeze(-1).unsqueeze(-1)
        cx = Kp[...,0,2].unsqueeze(-1).unsqueeze(-1)
        cy = Kp[...,1,2].unsqueeze(-1).unsqueeze(-1)
        d  = depths.unsqueeze(3)

        x_c = (uu - cx)/fx * d
        y_c = (vv - cy)/fy * d
        z_c = d
        ones = torch.ones_like(z_c)
        Xc = torch.cat([x_c,y_c,z_c,ones], dim=3)

        T = T_egofromcam.view(B,T,C,4,4,1,1)
        Xe = T[...,0:3,0:4,:,:] @ Xc.view(B,T,C,4,Hp*Wp).unsqueeze(-1)
        Xe = Xe.squeeze(-1).view(B,T,C,3,Hp,Wp)

        x = Xe[:,:,:,0]; y = Xe[:,:,:,1]
        valid = (d[:,:,:,0] > 0) & (x>=self.x_min) & (x<self.x_max) & (y>=self.y_min) & (y<self.y_max)
        ix = ((x - self.x_min)/rx).floor().clamp(0, self.Wb-1).long()
        iy = ((y - self.y_min)/ry).floor().clamp(0, self.Hb-1).long()
        lin = iy*self.Wb + ix

        N = B*T*C
        feats_flat = feats.reshape(N, Cr, Hp*Wp)
        lin_flat   = lin.reshape(N, Hp*Wp)
        valid_flat = valid.reshape(N, Hp*Wp)

        out = feats.new_zeros((N, Cr, self.Hb*self.Wb))
        cnt = feats.new_zeros((N, 1,  self.Hb*self.Wb))
        idx = lin_flat.masked_fill(~valid_flat, 0)
        out.scatter_add_(2, idx.unsqueeze(1).expand(-1, Cr, -1), feats_flat * valid_flat.unsqueeze(1))
        cnt.scatter_add_(2, idx.unsqueeze(1), torch.ones_like(feats_flat[:,:1]) * valid_flat.unsqueeze(1))

        out = out.view(B,T,C,Cr,self.Hb,self.Wb)
        cnt = cnt.view(B,T,C,1, self.Hb,self.Wb).clamp_min(1.0)
        return (out.sum(dim=2)/cnt.sum(dim=2))

class RGBLiftSplatEncoder(nn.Module):
    def __init__(self, H_bev, W_bev, x_min, x_max, y_min, y_max, C_r=128):
        super().__init__()
        self.enc  = _ImageEncoder2D(out_ch=C_r)
        self.view = _LiftSplat(H_bev,W_bev,x_min,x_max,y_min,y_max)
    def forward(self, cam_imgs, cam_depths, cam_K_scaled, cam_T_cam_from_ego, H, W):
        B,T,C,Ch,Himg,Wimg = cam_imgs.shape
        x = cam_imgs.reshape(B*T*C, Ch, Himg, Wimg)
        f = self.enc(x)                         # (B*T*C, C_r, H', W')
        Cr, Hp, Wp = f.shape[1], f.shape[2], f.shape[3]
        f = f.view(B, T, C, Cr, Hp, Wp)
        F_bev = self.view(f, cam_depths, cam_K_scaled, cam_T_cam_from_ego, H, W)  # (B,T,C_r,Hb,Wb)
        return F_bev
