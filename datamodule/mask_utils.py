# mask_utils.py — imports
from typing import Tuple, Optional
import numpy as np

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box


def T_global_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """
    global -> ego 4x4 transform at the timestamp of the given sample_data.
    """
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T_gego = Quaternion(ep["rotation"]).transformation_matrix
    T_gego[:3, 3] = np.asarray(ep["translation"], dtype=np.float32)
    return np.linalg.inv(T_gego).astype(np.float32)  # global -> ego


def xy_to_bev_px_float(self, x: float, y: float) -> Tuple[float, float]:
    ix = (x - self.bev_x_min) / self.res_x
    iy = (y - self.bev_y_min) / self.res_y
    ix = float(np.clip(ix, 0.0, self.W_bev - 1.0))
    iy = float(np.clip(iy, 0.0, self.H_bev - 1.0))
    return iy, ix

# ---- BEV pixel mapping helpers ----
def xy_to_bev_px(self, x: float, y: float) -> Tuple[int, int]:
    """Map ego (x,y) meters → (iy, ix) BEV pixels with y_min→top, x_min→left."""
    ix = int(np.floor((x - self.bev_x_min) / self.res_x))
    iy = int(np.floor((y - self.bev_y_min) / self.res_y))
    ix = np.clip(ix, 0, self.W_bev - 1)
    iy = np.clip(iy, 0, self.H_bev - 1)
    return int(iy), int(ix)

def order_poly_clockwise(self, P: np.ndarray) -> np.ndarray:
    c = P.mean(axis=0)
    ang = np.arctan2(P[:,1] - c[1], P[:,0] - c[0])
    return P[np.argsort(ang)]

def rasterize_convex_poly_bev(self, verts_iyix: np.ndarray) -> np.ndarray:
    xs = verts_iyix[:,1]; ys = verts_iyix[:,0]
    xmin = int(max(0, np.floor(xs.min())))
    xmax = int(min(self.W_bev - 1, np.ceil(xs.max())))
    ymin = int(max(0, np.floor(ys.min())))
    ymax = int(min(self.H_bev - 1, np.ceil(ys.max())))
    if xmax < xmin or ymax < ymin:
        m = np.zeros((self.H_bev, self.W_bev), dtype=np.uint8)
        return m
    gx, gy = np.meshgrid(np.arange(xmin, xmax + 1, dtype=np.float32),
                         np.arange(ymin, ymax + 1, dtype=np.float32))
    area = 0.0
    n = len(verts_iyix)
    for i in range(n):
        x0, y0 = verts_iyix[i,1], verts_iyix[i,0]
        x1, y1 = verts_iyix[(i+1)%n,1], verts_iyix[(i+1)%n,0]
        area += (x0 * y1 - x1 * y0)
    cw = (area < 0.0)
    inside = np.ones_like(gx, dtype=bool)
    for i in range(n):
        x0, y0 = verts_iyix[i,1], verts_iyix[i,0]
        x1, y1 = verts_iyix[(i+1)%n,1], verts_iyix[(i+1)%n,0]
        cross = (gx - x0) * (y1 - y0) - (gy - y0) * (x1 - x0)
        if cw:
            inside &= (cross <= 0.0)
        else:
            inside &= (cross >= 0.0)
    m = np.zeros((self.H_bev, self.W_bev), dtype=np.uint8)
    m[ymin:ymax+1, xmin:xmax+1] = inside.astype(np.uint8)
    return m

def ann_box_poly_ego_xy(self, ann: dict, lidar_sd: str) -> Optional[np.ndarray]:
    T_ge = T_global_to_ego_4x4(self.nusc, lidar_sd)
    box = Box(center=ann["translation"], size=ann["size"], orientation=Quaternion(ann["rotation"]))
    Cg = box.corners().T.astype(np.float32)
    Cg = np.hstack([Cg, np.ones((8,1), dtype=np.float32)])
    Ce = (T_ge @ Cg.T).T[:, :2]
    Ce_r = np.round(Ce, 3)
    uniq, idx = np.unique(Ce_r, axis=0, return_index=True)
    if len(idx) < 3:
        return None
    P = Ce[idx[:4]] if len(idx) >= 4 else Ce[idx]
    P = order_poly_clockwise(self, P)
    return P

def bev_mask_from_ann(self, ann: dict, lidar_sd: str) -> Tuple[np.ndarray, Tuple[int,int]]:
    P = ann_box_poly_ego_xy(self, ann, lidar_sd)
    if P is None:
        ctr_g = np.array(ann["translation"], dtype=np.float32)
        T_ge = T_global_to_ego_4x4(self.nusc, lidar_sd)
        ctr_e = (T_ge @ np.array([ctr_g[0], ctr_g[1], ctr_g[2], 1.0], dtype=np.float32))[:3]
        iy, ix = xy_to_bev_px(self, float(ctr_e[0]), float(ctr_e[1]))
        iy, ix = int(round(iy)), int(round(ix))
        m = np.zeros((self.H_bev, self.W_bev), dtype=np.uint8)
        m[iy, ix] = 1
        return m, (iy, ix)
    verts_iyix = np.stack([*zip(*[xy_to_bev_px_float(self, x, y) for x, y in P])], axis=-1)
    verts_iyix = np.array([[iy, ix] for iy, ix in verts_iyix], dtype=np.float32)
    mask = rasterize_convex_poly_bev(self, verts_iyix)
    ctr = np.mean(P, axis=0)
    ciy, cix = xy_to_bev_px_float(self, float(ctr[0]), float(ctr[1]))
    return mask, (int(round(ciy)), int(round(cix)))
