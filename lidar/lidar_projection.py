import json
import numpy as np
import torch

def _to_hom(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2 and X.shape[0] == 3:
        ones = np.ones((1, X.shape[1]), dtype=X.dtype)
        return np.vstack([X, ones])
    if X.ndim == 2 and X.shape[1] == 3:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])
    return X

def project_points(points_xyz_i: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray,
                   img_w: int, img_h: int, intensity_scale: float = 255.0):
    pts = points_xyz_i[:, :3].T
    inten = points_xyz_i[:, 3] if points_xyz_i.shape[1] > 3 else np.ones((pts.shape[1],), dtype=pts.dtype)
    h_up = pts[2].copy()                                  # CORRECTION: Z-up from LiDAR frame

    pc_cam = (T_cam_lidar @ _to_hom(pts))[:3]
    z = pc_cam[2]

    eps = 1e-6
    finite_xyz = np.isfinite(pc_cam).all(axis=0)
    valid_z = (z > eps) & np.isfinite(z)
    keep = finite_xyz & valid_z
    if not np.any(keep):
        empty_i32 = np.array([], np.int32)
        empty_f32 = np.array([], np.float32)
        return empty_i32, empty_i32, empty_f32, empty_f32, empty_f32

    pc_cam = pc_cam[:, keep]
    z = z[keep]
    inten = inten[keep]
    h_up = h_up[keep]                                      # CORRECTION

    uvw = K @ pc_cam
    u_f = np.divide(uvw[0], z, out=np.full_like(z, np.nan), where=z > eps)
    v_f = np.divide(uvw[1], z, out=np.full_like(z, np.nan), where=z > eps)
    u_r = np.rint(u_f); v_r = np.rint(v_f)

    ok = np.isfinite(u_r) & np.isfinite(v_r)
    if not np.any(ok):
        empty_i32 = np.array([], np.int32)
        empty_f32 = np.array([], np.float32)
        return empty_i32, empty_i32, empty_f32, empty_f32, empty_f32

    u_r = u_r[ok]; v_r = v_r[ok]
    z   = z[ok]
    inten = inten[ok]
    h_up = h_up[ok]                                        # CORRECTION

    inb = (u_r >= 0) & (u_r < img_w) & (v_r >= 0) & (v_r < img_h)
    if not np.any(inb):
        empty_i32 = np.array([], np.int32)
        empty_f32 = np.array([], np.float32)
        return empty_i32, empty_i32, empty_f32, empty_f32, empty_f32

    u = u_r[inb].astype(np.int32)
    v = v_r[inb].astype(np.int32)
    z = z[inb].astype(np.float32)
    inten = (inten[inb].astype(np.float32) / max(1e-6, intensity_scale))  # CORRECTION: normalize if needed
    h_up = h_up[inb].astype(np.float32)                                    # CORRECTION

    return u, v, z, inten, h_up                                            # CORRECTION: return h_up (not y_cam)

def rasterize(u, v, z, intensity, h_up, H, W, max_depth=80.0, h_min=-2.0, h_max=4.0, count_clip=5):
    size = H * W
    if u.size == 0:
        zeros = np.zeros((H, W), dtype=np.float32)
        return np.stack([
            zeros, zeros, zeros, zeros, zeros
        ], axis=0).astype(np.float32)

    idx = v * W + u
    order = np.lexsort((z, idx))               # sort by pixel then depth
    idx_s = idx[order]
    z_s   = z[order]
    i_s   = intensity[order]
    h_s   = h_up[order]                        # CORRECTION

    first = np.ones_like(idx_s, dtype=bool)
    first[1:] = idx_s[1:] != idx_s[:-1]

    idx_f = idx_s[first]
    z_f   = z_s[first]
    i_f   = i_s[first]
    h_f   = h_s[first]                         # CORRECTION

    depth  = np.zeros(size, dtype=np.float32)
    inten  = np.zeros(size, dtype=np.float32)
    height = np.zeros(size, dtype=np.float32)
    cnt    = np.bincount(idx, minlength=size).astype(np.float32)

    depth[idx_f]  = z_f
    inten[idx_f]  = i_f
    height[idx_f] = h_f                        # CORRECTION: use Z-up, no negation

    depth  = depth.reshape(H, W)
    inten  = inten.reshape(H, W)
    height = height.reshape(H, W)
    cnt    = cnt.reshape(H, W)
    occ    = (cnt > 0).astype(np.float32)

    d_n = np.clip(depth / max_depth, 0.0, 1.0)
    h_n = np.clip((height - h_min) / max(1e-6, (h_max - h_min)), 0.0, 1.0)  # CORRECTION: normalize Z-up
    i_n = np.clip(inten, 0.0, 1.0)
    c_n = np.clip(cnt / count_clip, 0.0, 1.0)

    return np.stack([d_n, h_n, i_n, c_n, occ], axis=0).astype(np.float32)

def lidar_to_2d_maps(points_xyz_i: np.ndarray, K: np.ndarray, T_cam_lidar: np.ndarray, W: int, H: int,
                     max_depth=80.0, h_min=-2.0, h_max=4.0, count_clip=5, intensity_scale: float = 255.0) -> torch.Tensor:
    u, v, z, inten, h_up = project_points(points_xyz_i, K, T_cam_lidar, W, H, intensity_scale)  # CORRECTION
    maps = rasterize(u, v, z, inten, h_up, H, W, max_depth, h_min, h_max, count_clip)           # CORRECTION
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
