import json
import numpy as np
import torch

def _to_hom(X):
    if X.ndim == 2 and X.shape[0] == 3:
        ones = np.ones((1, X.shape[1]), dtype=X.dtype)
        return np.vstack([X, ones])
    if X.ndim == 2 and X.shape[1] == 3:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([X, ones])
    return X

def project_points(points_xyz_i, K, T_cam_lidar, img_w, img_h):
    pts = points_xyz_i[:, :3].T
    inten = points_xyz_i[:, 3:4].T if points_xyz_i.shape[1] > 3 else np.ones((1, pts.shape[1]), dtype=pts.dtype)
    pc_cam = (T_cam_lidar @ _to_hom(pts))[:3]
    z = pc_cam[2]
    mask = z > 0
    pc_cam = pc_cam[:, mask]
    z = z[mask]
    inten = inten[:, mask]
    uvw = K @ pc_cam
    u = (uvw[0] / z).round().astype(np.int32)
    v = (uvw[1] / z).round().astype(np.int32)
    valid = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return u[valid], v[valid], z[valid], inten.reshape(-1)[valid], pc_cam[1, valid]

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
    return np.stack([d_n, h_n, i_n, c_n, occ], axis=0).astype(np.float32)

def lidar_to_2d_maps(points_xyz_i, K, T_cam_lidar, W, H, max_depth=80.0, h_min=-2.0, h_max=4.0, count_clip=5):
    u, v, z, inten, y = project_points(points_xyz_i, K, T_cam_lidar, W, H)
    maps = rasterize(u, v, z, inten, y, H, W, max_depth, h_min, h_max, count_clip)
    return torch.from_numpy(maps)

def adjust_intrinsics(K, src_size, dst_size):
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

def pack_calib(K, T_cam_lidar):
    return json.dumps({"K": K.tolist(), "T_cam_lidar": T_cam_lidar.tolist()})

def unpack_calib(s):
    o = json.loads(s)
    return np.array(o["K"], dtype=np.float32), np.array(o["T_cam_lidar"], dtype=np.float32)
