from typing import List, Tuple, Optional, Callable
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

def T_ego_to_global_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # Build ego->global homogeneous transform for a sample_data token
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T = np.eye(4, dtype=np.float32)
    R = np.asarray(Quaternion(ep["rotation"]).rotation_matrix, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(ep["translation"], dtype=np.float32)
    return T

def T_global_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # Invert to get global->ego
    return np.linalg.inv(T_ego_to_global_4x4(nusc, sd_token)).astype(np.float32)

def aggregate_sweeps_to_anchor(
    nusc: NuScenes,
    sweep_sd_tokens: List[str],
    key_sd_token: str,
    include_dt: bool,
    get_points_ego: Callable[[str], np.ndarray],  # returns (N,4) [x,y,z,intensity] in that sweep's ego
    dt_s: Optional[List[float]] = None,           # per-sweep Δt (seconds) relative to key frame
) -> np.ndarray:
    # Transform all sweeps to the anchor ego frame and concatenate; optionally append Δt as a feature
    if dt_s is None:
        dt_s = [0.0] * len(sweep_sd_tokens)

    T_anchor_from_global = T_global_to_ego_4x4(nusc, key_sd_token)  # anchor global->ego@t
    out = []

    for sdl, d in zip(sweep_sd_tokens, dt_s):
        pts_ego = get_points_ego(sdl)                                 # (N,4) in that sweep's ego
        T_global_from_sweep_ego = T_ego_to_global_4x4(nusc, sdl)      # sweep ego->global
        T_anchor_from_sweep_ego = T_anchor_from_global @ T_global_from_sweep_ego  # sweep ego->anchor ego

        # Apply rigid transform to xyz only
        xyz = (T_anchor_from_sweep_ego[:3, :3] @ pts_ego[:, :3].T + T_anchor_from_sweep_ego[:3, 3:4]).T

        if include_dt:
            f = np.concatenate(
                [xyz, pts_ego[:, 3:4], np.full((xyz.shape[0], 1), float(d), dtype=np.float32)],
                axis=1
            )  # [x,y,z,intensity,dt]
        else:
            f = np.concatenate([xyz, pts_ego[:, 3:4]], axis=1)        # [x,y,z,intensity]
        out.append(f)

    if not out:
        return np.zeros((0, 5 if include_dt else 4), dtype=np.float32)

    return np.vstack(out).astype(np.float32)                           # (ΣN, C)

def pillarize_points_xy(
    points_xyzit: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    vx: float, vy: float,                       # pillar size (XY)
    max_points_per_pillar: int,
    max_pillars: int,
    include_dt: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Convert raw points → PointPillars inputs:
    #  - feats: (P, M, C_feat), coords: (P, 2) [iy, ix], npoints: (P,)
    feat_dim = 10 if include_dt else 9

    if points_xyzit.shape[0] == 0:
        return (np.zeros((0, max_points_per_pillar, feat_dim), dtype=np.float32),
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.int32))

    # Crop to ROI including Z bounds
    mask = (
        (points_xyzit[:, 0] >= x_min) & (points_xyzit[:, 0] < x_max) &
        (points_xyzit[:, 1] >= y_min) & (points_xyzit[:, 1] < y_max) &
        (points_xyzit[:, 2] >= z_min) & (points_xyzit[:, 2] < z_max)
    )
    pts = points_xyzit[mask]
    if pts.shape[0] == 0:
        return (np.zeros((0, max_points_per_pillar, feat_dim), dtype=np.float32),
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.int32))

    # Discretize XY into pillar indices
    ix = np.floor((pts[:, 0] - x_min) / vx).astype(np.int32)
    iy = np.floor((pts[:, 1] - y_min) / vy).astype(np.int32)
    W = int(np.floor((x_max - x_min) / vx))
    pid = iy * W + ix                                             # flat pillar id

    # Stable sort by pillar id, then group
    order = np.argsort(pid, kind="mergesort")
    pid_sorted = pid[order]
    pts_sorted = pts[order]

    uniq, start_idx, counts = np.unique(pid_sorted, return_index=True, return_counts=True)
    K = min(max_pillars, uniq.shape[0])

    feats = np.zeros((K, max_points_per_pillar, feat_dim), dtype=np.float32)
    coords = np.zeros((K, 2), dtype=np.int32)
    npoints = np.zeros((K,), dtype=np.int32)

    for k in range(K):
        s = start_idx[k]; e = s + counts[k]
        pts_k = pts_sorted[s:e]

        iy_k = iy[order[s]]
        ix_k = ix[order[s]]

        m = min(max_points_per_pillar, pts_k.shape[0])
        chosen = pts_k[:m]

        # Per-point base features
        x = chosen[:, 0]; y = chosen[:, 1]; z = chosen[:, 2]; i = chosen[:, 3]
        # Pillar center in world coords (XY)
        xc = x_min + (ix_k + 0.5) * vx
        yc = y_min + (iy_k + 0.5) * vy
        # Cluster means
        x_mean = x.mean(); y_mean = y.mean(); z_mean = z.mean()

        # Assemble standard PP feature set
        f = [x, y, z, i, x - x_mean, y - y_mean, z - z_mean, x - xc, y - yc]
        if include_dt:
            f.append(chosen[:, 4])                                 # Δt if present
        f = np.stack(f, axis=1)                                    # (m, C_feat)

        if m < max_points_per_pillar:
            pad = np.zeros((max_points_per_pillar - m, f.shape[1]), dtype=np.float32)
            f = np.concatenate([f, pad], axis=0)                   # zero-pad to M

        feats[k] = f
        coords[k] = np.array([iy_k, ix_k], dtype=np.int32)         # (iy, ix)
        npoints[k] = m

    return feats, coords, npoints
