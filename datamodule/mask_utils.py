# --- put this in mask_utils.py (uses your loader's bev spec) ---
import numpy as np
import cv2
from pyquaternion import Quaternion

def _T_from_quat_trans(qwxyz, t_xyz):
    q = Quaternion(w=qwxyz[0], x=qwxyz[1], y=qwxyz[2], z=qwxyz[3])
    R = q.rotation_matrix
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(t_xyz, dtype=np.float32)
    return T

def _yaw_from_quat(qwxyz) -> float:
    q = Quaternion(w=qwxyz[0], x=qwxyz[1], y=qwxyz[2], z=qwxyz[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return float(yaw)

def _ego_from_global_T(nusc, lidar_sd_token):
    sd = nusc.get("sample_data", lidar_sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T_global_from_ego = _T_from_quat_trans(ep["rotation"], ep["translation"])
    T_ego_from_global = np.linalg.inv(T_global_from_ego).astype(np.float32)
    yaw_ego_global = _yaw_from_quat(ep["rotation"])
    return T_ego_from_global, yaw_ego_global

def _rect_corners_xy(center_xy, yaw, length, width):
    cx, cy = center_xy
    hl, hw = 0.5*float(length), 0.5*float(width)
    local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float32)  # (4,2)

def _xy_to_bev_pixels(xs, ys, x_min, y_min, res_x, res_y):
    cols = (xs - x_min) / float(res_x)  # x -> col
    rows = (ys - y_min) / float(res_y)  # y -> row (y_min at top)
    return rows, cols

def ann_to_bev_polygon_pixels(loader, ann, lidar_sd_token):
    """
    Returns a (4,2) int32 array of (x=col, y=row) pixel corners for the box polygon
    in the loader's BEV grid. Returns None if ann is None.
    """
    if ann is None:
        return None

    # global -> ego@LiDAR time
    T_ego_from_global, yaw_ego_global = _ego_from_global_T(loader.nusc, lidar_sd_token)

    # center in ego
    t_g = np.array(ann["translation"], dtype=np.float32)  # (x,y,z)
    p_g = np.array([t_g[0], t_g[1], t_g[2], 1.0], dtype=np.float32)
    p_e = T_ego_from_global @ p_g
    cx_e, cy_e = float(p_e[0]), float(p_e[1])

    # nuScenes size = [width, length, height]
    size = np.asarray(ann["size"], dtype=np.float32)
    width, length = float(size[0]), float(size[1])

    # yaw in ego
    yaw_e = _yaw_from_quat(ann["rotation"]) - yaw_ego_global

    # 4 corners in ego meters
    xy_e = _rect_corners_xy((cx_e, cy_e), yaw_e, length=length, width=width)  # (4,2)

    # meters -> pixels (your convention)
    r, c = _xy_to_bev_pixels(
        xs=xy_e[:,0], ys=xy_e[:,1],
        x_min=loader.bev_x_min, y_min=loader.bev_y_min,
        res_x=loader.res_x,     res_y=loader.res_y
    )
    poly = np.stack([c, r], axis=1)  # (x,y) = (col,row)
    poly = np.rint(poly).astype(np.int32)

    # clamp to canvas
    poly[:,0] = np.clip(poly[:,0], 0, loader.W_bev - 1)
    poly[:,1] = np.clip(poly[:,1], 0, loader.H_bev - 1)
    return poly  # (4,2) int32

def bev_box_mask_from_ann(loader, ann, lidar_sd_token, fill_value=1):
    """
    Rasterize the oriented box polygon into a (H_bev, W_bev) uint8 mask.
    """
    H, W = loader.H_bev, loader.W_bev
    mask = np.zeros((H, W), dtype=np.uint8)
    poly = ann_to_bev_polygon_pixels(loader, ann, lidar_sd_token)
    if poly is None:
        return mask
    cv2.fillPoly(mask, [poly], int(fill_value))
    return mask
