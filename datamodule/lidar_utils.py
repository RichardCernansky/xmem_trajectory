import numpy as np
from typing import Tuple
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

def pose_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """
    Ego pose (ego -> global) at the timestamp of a given sample_data token.
    Returns T_global_from_ego (4x4, float32).
    """
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T = Quaternion(ep["rotation"]).transformation_matrix  # rotation (ego->global)
    T[:3, 3] = np.asarray(ep["translation"], dtype=np.float32)  # translation in global
    return T.astype(np.float32)

def T_sensor_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """
    Calibrated sensor extrinsic: sensor -> ego (4x4, float32).
    """
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T = Quaternion(cs["rotation"]).transformation_matrix  # rotation (sensor->ego)
    T[:3, 3] = np.asarray(cs["translation"], dtype=np.float32)  # translation in ego
    return T.astype(np.float32)

def T_ego_to_sensor_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """
    Inverse of calibrated extrinsic: ego -> sensor (4x4, float32).
    """
    return np.linalg.inv(T_sensor_to_ego_4x4(nusc, sd_token)).astype(np.float32)


def lidar_points_ego(self, lidar_sd_token: str) -> np.ndarray:
        """LiDAR sweep → ego frame @ that timestamp. Returns (N,4): [x,y,z,intensity]."""
        path = self.nusc.get_sample_data_path(lidar_sd_token)
        pc = LidarPointCloud.from_file(path).points  # (4,N)
        xyz = pc[:3, :].T.astype(np.float32)
        inten = pc[3, :].astype(np.float32)

        # normalize intensity if it looks like 0..255
        max_i = inten.max() if inten.size else 0.0
        if max_i > 1.5:
            inten = inten / 255.0

        T_sensor_from_ego = T_sensor_from_ego_4x4(self.nusc, lidar_sd_token)     # sensor <- ego
        T_ego_from_sensor = np.linalg.inv(T_sensor_from_ego).astype(np.float32)  # ego <- sensor
        xyz_ego = transform_points(T_ego_from_sensor, xyz)
        return np.concatenate([xyz_ego, inten[:, None]], axis=1).astype(np.float32)

def lidar_bev_from_points_fixed(self, pts_ego: np.ndarray) -> np.ndarray:
    """
    Aggregate ego-frame points into a FIXED BEV grid: (H_bev, W_bev),
    channels = [log1p(count), mean_z, max_z, mean_intensity].
    Mapping: y_min -> top rows, y_max -> bottom rows (image rows increase downward).
    """
    C = 4
    bev = np.zeros((C, self.H_bev, self.W_bev), dtype=np.float32)
    if pts_ego.size == 0:
        return bev

    x = pts_ego[:, 0]; y = pts_ego[:, 1]; z = pts_ego[:, 2]; inten = pts_ego[:, 3]

    # ROI filter
    m = (x >= self.bev_x_min) & (x < self.bev_x_max) & (y >= self.bev_y_min) & (y < self.bev_y_max)
    if not np.any(m):
        return bev
    x = x[m]; y = y[m]; z = z[m]; inten = inten[m]

    # Discretize with inferred meters-per-pixel
    ix = np.floor((x - self.bev_x_min) / self.res_x).astype(np.int64)  # [0..W_bev-1]
    iy = np.floor((y - self.bev_y_min) / self.res_y).astype(np.int64)  # [0..H_bev-1]
    np.clip(ix, 0, self.W_bev - 1, out=ix)
    np.clip(iy, 0, self.H_bev - 1, out=iy)
    lin = iy * self.W_bev + ix  # flattened indices

    HW = self.H_bev * self.W_bev
    count = np.bincount(lin, minlength=HW).astype(np.float32)
    sum_z = np.bincount(lin, weights=z, minlength=HW).astype(np.float32)
    sum_i = np.bincount(lin, weights=inten, minlength=HW).astype(np.float32)
    max_z = np.full(HW, -np.inf, dtype=np.float32)
    np.maximum.at(max_z, lin, z)

    eps = 1e-6
    mean_z = sum_z / (count + eps)
    mean_i = sum_i / (count + eps)

    # reshape & stabilize
    count = count.reshape(self.H_bev, self.W_bev)
    mean_z = mean_z.reshape(self.H_bev, self.W_bev)
    max_z  = max_z.reshape(self.H_bev, self.W_bev)
    mean_i = mean_i.reshape(self.H_bev, self.W_bev)

    mean_z = np.clip(mean_z, -3.0, 5.0)
    max_z  = np.clip(max_z,  -3.0, 5.0)
    mean_i = np.clip(mean_i,  0.0, 1.0)

    bev[0] = np.log1p(count)
    bev[1] = mean_z
    bev[2] = max_z
    bev[3] = mean_i
    return bev

def T_sensor_from_ego_4x4(nusc: NuScenes, cam_sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])  # sensor→ego
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.asarray(cs["translation"], dtype=np.float32)
    T_ego_from_cam = np.eye(4, dtype=np.float32)
    T_ego_from_cam[:3, :3] = R
    T_ego_from_cam[:3,  3] = t
    return np.linalg.inv(T_ego_from_cam).astype(np.float32)  # cam←ego

def T_cam_from_lidar_4x4(nusc, cam_sd_token, lidar_sd_token) -> np.ndarray:
    # cam←ego_cam
    T_cam_from_ego = T_sensor_from_ego_4x4(nusc, cam_sd_token)            # ✅ sensor←ego
    # ego_cam←global(cam-time)
    T_ego_cam_from_global = np.linalg.inv(pose_4x4(nusc, cam_sd_token))   # ✅
    # global←ego_lidar(lidar-time)
    T_global_from_ego_lidar = pose_4x4(nusc, lidar_sd_token)              # ✅
    # ego_lidar←lidar
    T_ego_from_lidar = T_sensor_to_ego_4x4(nusc, lidar_sd_token)          # ✅
    return (T_cam_from_ego @ T_ego_cam_from_global @
            T_global_from_ego_lidar @ T_ego_from_lidar).astype(np.float32)


def scale_K(K: np.ndarray, orig_hw: Tuple[int,int], target_hw: Tuple[int,int]) -> np.ndarray:
    # Unpack sizes: (H,W) original → (H,W) target
    oh, ow = orig_hw; th, tw = target_hw
    # Compute independent x/y scale factors
    sx, sy = tw/float(ow), th/float(oh)
    # Copy so we don’t mutate the input
    K2 = K.copy()
    # Scale focal lengths
    K2[0,0] *= sx; K2[1,1] *= sy
    # Scale principal point
    K2[0,2] *= sx; K2[1,2] *= sy
    # Return intrinsics tailored to the resized crop
    return K2

def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    # Append homogeneous 1 to each point
    P = np.concatenate([pts_xyz, np.ones((pts_xyz.shape[0],1), np.float32)], axis=1)
    # Apply 4x4 transform, drop homogeneous
    X = (T @ P.T).T[:, :3]
    # Return transformed 3D points
    return X

def rasterize_depth_xyz_cam(xyz_cam: np.ndarray, K: np.ndarray, hw: Tuple[int,int]) -> np.ndarray:
    H, W = hw

    # PATCH: require finite XYZ and positive z; use small epsilon to avoid near-zero z
    z_all = xyz_cam[:, 2]
    finite_xyz = np.isfinite(xyz_cam).all(axis=1)                  # PATCH
    m = (z_all > 1e-9) & finite_xyz                                # PATCH
    if not np.any(m):
        return np.zeros((H, W), dtype=np.float32)

    x = xyz_cam[m, 0]
    y = xyz_cam[m, 1]
    z = z_all[m]

    # PATCH: safe projection using np.divide (avoids NaN/Inf creation at source)
    u_f = np.divide(K[0, 0] * x, z, out=np.full_like(z, np.nan), where=z > 1e-9) + K[0, 2]   # PATCH
    v_f = np.divide(K[1, 1] * y, z, out=np.full_like(z, np.nan), where=z > 1e-9) + K[1, 2]   # PATCH

    # PATCH: round without casting yet; then drop non-finite results
    u_r = np.rint(u_f)                                                                        # PATCH
    v_r = np.rint(v_f)                                                                        # PATCH
    good = np.isfinite(u_r) & np.isfinite(v_r)                                                # PATCH
    if not np.any(good):
        return np.zeros((H, W), dtype=np.float32)
    u_r = u_r[good]; v_r = v_r[good]; z = z[good]                                             # PATCH

    # PATCH: in-bounds check still on float; cast only after this filter
    m2 = (u_r >= 0) & (u_r < W) & (v_r >= 0) & (v_r < H)                                      # PATCH
    if not np.any(m2):
        return np.zeros((H, W), dtype=np.float32)
    u = u_r[m2].astype(np.int32)                                                              # PATCH
    v = v_r[m2].astype(np.int32)                                                              # PATCH
    z = z[m2]

    # Initialize depth image with +inf so we can do a min-z z-buffer
    depth = np.full((H, W), np.inf, dtype=np.float32)

    # Flatten pixel coords for fast grouping
    flat = v * W + u
    order = np.argsort(flat)
    flat = flat[order]; z = z[order]

    # Walk runs of same pixel; keep nearest z per pixel
    prev = None; best = None
    for f, zv in zip(flat, z):
        if prev is None:
            prev = f; best = zv
        elif f == prev:
            if zv < best: best = zv
        else:
            r, c = divmod(prev, W); depth[r, c] = best
            prev = f; best = zv
    if prev is not None:
        r, c = divmod(prev, W); depth[r, c] = best

    # Replace untouched pixels (inf) by 0.0 = "no hit"
    depth[~np.isfinite(depth)] = 0.0
    return depth


def compose_three_depth(dL: np.ndarray, dF: np.ndarray, dR: np.ndarray,
                        cw: int, ov_lf_px: int, ov_fr_px: int) -> np.ndarray:
    # Panorama size
    H = dL.shape[0]
    W = 3*cw - (ov_lf_px + ov_fr_px)
    out = np.zeros((H, W), dtype=np.float32)

    # 1) Left no-overlap region: [0, cw - ov_lf)
    out[:, :cw - ov_lf_px] = dL[:, :cw - ov_lf_px]

    # 2) Left–Front overlap: [cw - ov_lf, cw)
    if ov_lf_px > 0:
        L_ov  = dL[:, cw - ov_lf_px : cw]          # width = ov_lf
        F_ovL = dF[:, :ov_lf_px]                   # width = ov_lf
        mixLF = np.where((L_ov==0) | ((F_ovL>0) & (F_ovL < L_ov)), F_ovL, L_ov)
        out[:, cw - ov_lf_px : cw] = mixLF

    # 3) Front no-overlap region (cut both overlaps) goes at:
    #    [cw, cw + (cw - ov_lf - ov_fr))
    mid_start = cw
    mid_end   = cw + (cw - ov_lf_px - ov_fr_px)
    if mid_end > mid_start:
        out[:, mid_start:mid_end] = dF[:, ov_lf_px : cw - ov_fr_px]  # width matches exactly

    # 4) Front–Right overlap: [mid_end, mid_end + ov_fr)
    if ov_fr_px > 0:
        F_ovR = dF[:, cw - ov_fr_px : cw]          # width = ov_fr
        R_ovF = dR[:, :ov_fr_px]                   # width = ov_fr
        mixFR = np.where((F_ovR==0) | ((R_ovF>0) & (R_ovF < F_ovR)), R_ovF, F_ovR)
        out[:, mid_end : mid_end + ov_fr_px] = mixFR

    # 5) Right no-overlap region: [mid_end + ov_fr, end)
    right_start = mid_end + ov_fr_px
    if right_start < W:
        out[:, right_start:] = dR[:, ov_fr_px:]

    return out

def invdepth_valid_from_depth(depth_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth_max_m = 80.0
    depth_min_m = 0.3

    # depth_m: (H,W), meters; 0.0 = no hit
    eps = 1e-9
    valid = (depth_m > 0.0).astype(np.float32)
    d = depth_m.copy()
    d[d <= 0.0] = np.inf  # avoid 1/0; invalid stays invalid via 'valid' mask
    inv = 1.0 / d
    inv_min = 1.0 / max(depth_max_m, eps)
    inv_max = 1.0 / max(depth_min_m, eps)
    inv01 = (inv - inv_min) / max(inv_max - inv_min, eps)
    inv01 = np.clip(inv01, 0.0, 1.0)
    inv01[valid == 0.0] = 0.0  # keep invalid pixels at 0
    return inv01.astype(np.float32), valid.astype(np.float32)