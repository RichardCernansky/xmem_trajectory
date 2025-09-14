import json
import numpy as np
import torch

# -----------------------------------------------------------------------------
# LiDAR → Image Projection: What this code does (step-by-step)
# -----------------------------------------------------------------------------
# 1) Make points homogeneous (_to_hom)
#    - Input points are shape (N, 3) or (3, N).
#    - Append a 1 to each point → shape becomes (N, 4) or (4, N) so a single 4×4
#      matrix multiply works.
#
# 2) Transform LiDAR → camera frame (project_points)
#    - Use the rigid transform T_cam_lidar ∈ R^{4×4}:
#        X_cam = T_cam_lidar @ [X_lidar; 1]
#    - Keep only points with positive depth (z > 0) in the camera frame.
#
# 3) Project into pixel coordinates with intrinsics K (project_points)
#    - Intrinsics K ∈ R^{3×3} map camera coords to the image plane:
#        [u'; v'; w'] = K @ [x; y; z]
#        u = round(u' / z),  v = round(v' / z)
#    - Discard pixels outside the image bounds [0, W) × [0, H).
#
# 4) Rasterize per-pixel statistics (rasterize)
#    - Given all valid hits (u, v) with depth z, camera-Y y_cam, and intensity:
#        • Depth:     keep minimum depth per pixel         (np.minimum.at(depth, idx, z))
#        • Height:    average y_cam per pixel
#        • Intensity: average reflectivity per pixel
#        • Count:     number of hits (optionally clipped)
#        • Occupancy: 1 if any hit occurred
#
# 5) (Elsewhere) Normalize channels to [0, 1] and stack into a 5-channel map:
#    [depth_norm, height_norm, intensity_norm, count_norm, occupancy].
#    These maps are pixel-aligned with RGB and ready for early fusion.
# -----------------------------------------------------------------------------


def _to_hom(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2 and X.shape[0] == 3:
        ones = np.ones((1, X.shape[1]), dtype=X.dtype)
        return np.vstack([X, ones])
    if X.ndim == 2 and X.shape[1] == 3:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])
    return X

def project_points(points_xyz_i: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray, img_w: int, img_h: int):
    # Mulitple masking to check for NaN/Inf values

    pts = points_xyz_i[:, :3].T
    inten = points_xyz_i[:, 3:4].T if points_xyz_i.shape[1] > 3 else np.ones((1, pts.shape[1]), dtype=pts.dtype)

    # LiDAR -> camera
    pc_cam = (T_cam_lidar @ _to_hom(pts))[:3]            # shape (3, N)
    z = pc_cam[2]

    # Guard everything: positive depth + all finite components
    eps = 1e-6
    finite_xyz = np.isfinite(pc_cam).all(axis=0)
    valid_z = (z > eps) & np.isfinite(z)
    keep0 = finite_xyz & valid_z
    if not np.any(keep0):
        return np.array([], np.int32), np.array([], np.int32), np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    pc_cam = pc_cam[:, keep0]
    z = z[keep0]
    inten = inten[:, keep0]

    # Intrinsics projection
    uvw = K @ pc_cam

    # Safe division (float), no cast yet
    u_f = np.divide(uvw[0], z, out=np.full_like(z, np.nan), where=z > eps)
    v_f = np.divide(uvw[1], z, out=np.full_like(z, np.nan), where=z > eps)

    # Round to nearest pixel as float
    u_r = np.rint(u_f)
    v_r = np.rint(v_f)

    # Keep only finite & in-bounds pixels BEFORE casting
    finite_uv = np.isfinite(u_r) & np.isfinite(v_r)
    if not np.any(finite_uv):
        return np.array([], np.int32), np.array([], np.int32), np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    u_r = u_r[finite_uv]
    v_r = v_r[finite_uv]
    z   = z[finite_uv]
    inten = inten[:, finite_uv]
    y_cam = pc_cam[1, finite_uv]

    inb = (u_r >= 0) & (u_r < img_w) & (v_r >= 0) & (v_r < img_h)
    if not np.any(inb):
        return np.array([], np.int32), np.array([], np.int32), np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    # Finally cast after filtering
    u = u_r[inb].astype(np.int32)
    v = v_r[inb].astype(np.int32)
    z = z[inb].astype(np.float32)
    inten = inten.reshape(-1)[inb].astype(np.float32)
    y_cam = y_cam[inb].astype(np.float32)

    return u, v, z, inten, y_cam


def rasterize(u, v, z, intensity, y_cam, H, W, max_depth=80.0, h_min=-2.0, h_max=4.0, count_clip=5):
    depth = np.full((H, W), np.inf, dtype=np.float32)
    sum_h = np.zeros((H, W), dtype=np.float32)
    sum_i = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    idx = v * W + u
    np.minimum.at(depth.reshape(-1), idx, z)
    np.add.at(sum_h.reshape(-1), idx, y_cam)
    np.add.at(sum_i.reshape(-1), idx, intensity)
    np.add.at(cnt.reshape(-1), idx, 1.0)
    occ = (cnt > 0).astype(np.float32)
    depth[np.isinf(depth)] = 0.0
    d_n = np.clip(depth / max_depth, 0.0, 1.0)
    avg_h = np.zeros_like(sum_h)
    m = cnt > 0
    avg_h[m] = sum_h[m] / cnt[m]
    h_n = np.clip((avg_h - h_min) / max(1e-6, (h_max - h_min)), 0.0, 1.0)
    avg_i = np.zeros_like(sum_i)
    avg_i[m] = sum_i[m] / cnt[m]
    i_n = np.clip(avg_i, 0.0, 1.0)
    c_n = np.clip(cnt / count_clip, 0.0, 1.0)
    return np.stack([d_n, h_n, i_n, c_n, occ], axis=0).astype(np.float32) # Shape: [5, H, W] = [depth, height, intensity, count, occupancy].

def lidar_to_2d_maps(points_xyz_i: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray, W: int, H: int,
                     max_depth=80.0, h_min=-2.0, h_max=4.0, count_clip=5) -> torch.Tensor:
    u, v, z, inten, y = project_points(points_xyz_i, K, T_cam_lidar, W, H)
    maps = rasterize(u, v, z, inten, y, H, W, max_depth, h_min, h_max, count_clip)
    return torch.from_numpy(maps)

def adjust_intrinsics(K: np.ndarray, src_size, dst_size):
    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / sw
    sy = dh / sh
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2

def pack_calib(K: np.ndarray, T_cam_lidar: np.ndarray) -> str:
    return json.dumps({"K": K.tolist(), "T_cam_lidar": T_cam_lidar.tolist()})

def unpack_calib(s: str):
    o = json.loads(s)
    return np.array(o["K"], dtype=np.float32), np.array(o["T_cam_lidar"], dtype=np.float32)
