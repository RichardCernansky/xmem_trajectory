import numpy as np
from typing import Tuple
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


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

def pose_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # Load the sample_data record to reach its ego_pose
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    # Convert ego rotation (quaternion) into a 4x4 transform
    T = Quaternion(ep["rotation"]).transformation_matrix
    # Insert translation (x,y,z) into the matrix
    T[:3, 3] = np.asarray(ep["translation"], dtype=np.float32)
    # Return homogeneous transform: ego_in_global at that timestamp
    return T.astype(np.float32)


def T_sensor_from_ego_4x4(nusc: NuScenes, cam_sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])  # sensor→ego
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.asarray(cs["translation"], dtype=np.float32)
    T_ego_from_cam = np.eye(4, dtype=np.float32)
    T_ego_from_cam[:3, :3] = R
    T_ego_from_cam[:3,  3] = t
    return np.linalg.inv(T_ego_from_cam).astype(np.float32)  # cam←ego


def T_sensor_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # sample_data -> calibrated_sensor to get sensor↔ego calibration at that timestamp
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    # 4x4 rotation from sensor to ego
    T = Quaternion(cs["rotation"]).transformation_matrix
    # Add translation from sensor to ego
    T[:3, 3] = np.asarray(cs["translation"], dtype=np.float32)
    # sensor_in_ego transform
    return T.astype(np.float32)

def T_ego_to_sensor_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # Invert sensor->ego to get ego->sensor
    return np.linalg.inv(T_sensor_to_ego_4x4(nusc, sd_token)).astype(np.float32)

def T_cam_from_lidar_4x4(nusc: NuScenes, cam_sd_token: str, lidar_sd_token: str) -> np.ndarray:
    # cam_in_ego
    T_cam_from_ego = T_ego_to_sensor_4x4(nusc, cam_sd_token)
    # ego_in_cam timestamp’s global -> ego (inverse of ego_in_global at camera time)
    T_ego_cam_from_global = np.linalg.inv(pose_4x4(nusc, cam_sd_token)).astype(np.float32)
    # ego_in_global at LiDAR time
    T_global_from_ego_lidar = pose_4x4(nusc, lidar_sd_token)
    # lidar_in_ego
    T_ego_from_lidar = T_sensor_to_ego_4x4(nusc, lidar_sd_token)
    # Chain: cam←ego * ego←global(cam) * global←ego(lidar) * ego←lidar  == cam←lidar
    return (T_cam_from_ego @ T_ego_cam_from_global @ T_global_from_ego_lidar @ T_ego_from_lidar).astype(np.float32)

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

    # PATCH: also require finite XYZ; keep only points in front of camera with positive z
    z_all = xyz_cam[:, 2]
    finite_xyz = np.isfinite(xyz_cam).all(axis=1)                     # PATCH
    m_front = (z_all > 1e-9) & finite_xyz                             # PATCH
    if not np.any(m_front):
        return np.zeros((H, W), dtype=np.float32)

    x = xyz_cam[m_front, 0]
    y = xyz_cam[m_front, 1]
    z = z_all[m_front]

    # PATCH: safe projection using np.divide to avoid NaNs/Infs from tiny denominators
    u_f = np.divide(K[0, 0] * x, z, out=np.full_like(z, np.nan), where=z > 1e-9) + K[0, 2]   # PATCH
    v_f = np.divide(K[1, 1] * y, z, out=np.full_like(z, np.nan), where=z > 1e-9) + K[1, 2]   # PATCH

    # PATCH: drop any non-finite projections before rounding/casting
    good_proj = np.isfinite(u_f) & np.isfinite(v_f)                                          # PATCH
    if not np.any(good_proj):
        return np.zeros((H, W), dtype=np.float32)
    u_f = u_f[good_proj]; v_f = v_f[good_proj]; z = z[good_proj]                             # PATCH

    # Round to nearest pixel (now guaranteed finite)
    u = np.rint(u_f).astype(np.int32)
    v = np.rint(v_f).astype(np.int32)

    # Keep points that land inside the image bounds
    m_in = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(m_in):
        return np.zeros((H, W), dtype=np.float32)
    u = u[m_in]; v = v[m_in]; z = z[m_in]

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

