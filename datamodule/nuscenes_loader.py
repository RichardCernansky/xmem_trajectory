from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from trainer.utils import open_config
from data.configs.filenames import TRAIN_CONFIG
from .image_utils import load_resize_arr
from .mask_utils import *
from .lidar_utils import (
    T_cam_from_lidar_4x4,           # camera<-lidar (4x4)
    T_sensor_from_ego_4x4,          # sensor<-ego (4x4)
    transform_points,               # apply 4x4 to Nx3
    scale_K,                        # scale intrinsics to (H,cw)
    rasterize_depth_xyz_cam,        # z-buffer rasterization
    lidar_points_ego,               # (N,4) in ego: [x,y,z,intensity]
)
from .pillars_utils import aggregate_sweeps_to_anchor, pillarize_points_xy

def _ann_for_instance_at_sample(nusc: NuScenes, sample_token: str, instance_token: str) -> Optional[dict]:
    # Find this instance's annotation in the sample (or None if missing)
    s = nusc.get("sample", sample_token)
    for ann_tok in s["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        if ann["instance_token"] == instance_token:
            return ann
    return None

class NuScenesLoader(Dataset):
    """
    Returns:
      - Camera tensors (time-major)
      - PointPillars tensors per timestep:
          pillar_features:   (T, P, M, C_feat)
          pillar_coords:     (T, P, 2)      [iy, ix]
          pillar_num_points: (T, P)
      - Meta with grid size for BEV scatter
    """
    def __init__(self, nusc: NuScenes, rows: List[Dict[str, Any]], dtype: torch.dtype = torch.float32):
        cfg = open_config(TRAIN_CONFIG)

        # Core refs
        self.nusc = nusc
        self.rows = rows
        self.dtype = dtype
        self.data_root = Path(self.nusc.dataroot).resolve()

        # Image sizing
        self.H  = int(cfg.get("H", 400))
        self.cw = int(cfg.get("cw", 320))
        self.trip = cfg.get("trip")                                      # camera order (e.g., front-left, front, front-right)

        # BEV ROI (meters)
        bev_x_bounds = cfg.get("bev_x_bounds", [-5.0, 55.0])
        bev_y_bounds = cfg.get("bev_y_bounds", [-30.0, 30.0])
        self.bev_x_min, self.bev_x_max = map(float, bev_x_bounds)
        self.bev_y_min, self.bev_y_max = map(float, bev_y_bounds)

        # Fixed BEV grid for depth/visualization (kept for compatibility)
        vx, vy, _ = self.pp_voxel_size
        self.H_bev = int(np.floor((self.bev_y_max - self.bev_y_min) / vy))
        self.W_bev = int(np.floor((self.bev_x_max - self.bev_x_min) / vx))
        self.res_y = vy
        self.res_x = vx

        # PointPillars discretization config
        self.pp_voxel_size = tuple(cfg.get("pp_voxel_size", [0.16, 0.16, 6.0]))  # (vx, vy, vz[unused])
        self.pp_z_bounds   = tuple(cfg.get("pp_z_bounds",  [-3.0, 3.0]))         # (z_min, z_max)
        self.pp_max_points = int(cfg.get("pp_max_points", 32))                   # M
        self.pp_max_pillars= int(cfg.get("pp_max_pillars", 12000))               # P cap
        self.pp_include_dt = bool(cfg.get("pp_include_dt", True))                # add Δt to features

        # Cache for raw LiDAR xyz (for depth projection speed)
        self._lidar_xyz_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_path(self, p: str) -> str:
        # Resolve relative path (stored in index) to absolute
        q = Path(p)
        return str(q if q.is_absolute() else (self.data_root / q).resolve())

    def _load_lidar_xyz(self, sd_token: str) -> np.ndarray:
        # Cache raw LiDAR xyz for repeated projections (depth maps)
        if sd_token in self._lidar_xyz_cache:
            return self._lidar_xyz_cache[sd_token]
        xyz = LidarPointCloud.from_file(self.nusc.get_sample_data_path(sd_token)).points[:3, :].T.astype(np.float32)
        self._lidar_xyz_cache[sd_token] = xyz
        return xyz

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        cams_all: List[str] = row["cam_set"]
        cam_idx = [cams_all.index(c) for c in self.trip]                # indices in your camera triplet
        C = len(self.trip)
        T_in = len(row["obs_cam_img_grid"])                             # time steps per sample

        # Per-camera accumulators
        cam_imgs_t:   List[List[torch.Tensor]] = [[] for _ in range(C)]
        cam_Ks_t:     List[List[np.ndarray]]   = [[] for _ in range(C)]
        cam_Tce_t:    List[List[np.ndarray]]   = [[] for _ in range(C)]
        cam_depths_t: List[List[torch.Tensor]] = [[] for _ in range(C)]
        cam_valids_t: List[List[torch.Tensor]] = [[] for _ in range(C)]

        # Pillar accumulators
        pillar_feats_t:  List[np.ndarray] = []
        pillar_coords_t: List[np.ndarray] = []
        pillar_npts_t:   List[np.ndarray] = []

        # Supervision
        traj     = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)
        last_pos = torch.tensor(row["target"]["last_xy"],   dtype=self.dtype)
        inst_tok = row["target"]["agent_id"]
        bev_mask_list: List[torch.Tensor] = []

        for t in range(T_in):
            # ---- Camera images, intrinsics, extrinsics ----
            img_paths_rel = [row["obs_cam_img_grid"][t][cam_idx[i]] for i in range(C)]
            img_paths = [self.resolve_path(p) for p in img_paths_rel]

            ims, orig_hw = [], []
            for p in img_paths:
                im, (ow, oh) = load_resize_arr(self, p, self.cw, self.H)        # resize keeping aspect
                ims.append(im)
                orig_hw.append((oh, ow))                                        # (orig_h, orig_w)

            # Tokens for LiDAR sweeps at this time step
            lidar_sweeps: List[str] = row["lidar"]["sd_tokens"][t]              # list of sweep sd_tokens
            lidar_key: str = row["lidar"].get("keyframe_sd_tokens", [lidar_sweeps[0]])[t]  # anchor
            dt_list: List[float] = row["lidar"].get("dt_s", [[0.0]*len(lidar_sweeps)])[t]  # Δt per sweep

            # Camera tokens & intrinsics at this time step
            sd_cams = [row["cams"][self.trip[i]]["sd_tokens"][t] for i in range(C)]
            Ks_orig = [np.asarray(row["cams"][self.trip[i]]["intrinsics"], dtype=np.float32) for i in range(C)]

            # ---- Aggregate sweeps to anchor ego@t and pillarize ----
            pts_xyzit = aggregate_sweeps_to_anchor(
                self.nusc,
                lidar_sweeps,
                lidar_key,
                self.pp_include_dt,
                get_points_ego=lambda sd: lidar_points_ego(self, sd),           # (N,4) in that sweep's ego
                dt_s=dt_list
            )
            vx, vy, _ = self.pp_voxel_size
            z_min, z_max = self.pp_z_bounds

            feats_np, coords_np, npts_np = pillarize_points_xy(
                pts_xyzit,
                self.bev_x_min, self.bev_x_max,
                self.bev_y_min, self.bev_y_max,
                z_min, z_max,
                vx, vy,
                self.pp_max_points,
                self.pp_max_pillars,
                include_dt=self.pp_include_dt
            )
            pillar_feats_t.append(feats_np)                                      # (P, M, C_feat)
            pillar_coords_t.append(coords_np)                                    # (P, 2) [iy, ix]
            pillar_npts_t.append(npts_np)                                        # (P,)

            # ---- Build per-camera depth by projecting union of sweeps ----
            for i_cam in range(C):
                (oh, ow) = orig_hw[i_cam]
                K_scaled = scale_K(Ks_orig[i_cam], (oh, ow), (self.H, self.cw))  # rescaled intrinsics
                T_cam_from_ego = T_sensor_from_ego_4x4(self.nusc, sd_cams[i_cam])

                # Project every sweep to this camera at this step
                Xc_all = []
                for sdl in lidar_sweeps:
                    xyz_lidar = self._load_lidar_xyz(sdl)                        # cached (N,3)
                    T_cl = T_cam_from_lidar_4x4(self.nusc, sd_cams[i_cam], sdl)  # camera<-lidar
                    Xc_all.append(transform_points(T_cl, xyz_lidar))
                Xc_cat = np.vstack(Xc_all) if Xc_all else np.zeros((0, 3), dtype=np.float32)

                d = rasterize_depth_xyz_cam(Xc_cat, K_scaled, (self.H, self.cw)) # (H,cw) depth map

                cam_imgs_t[i_cam].append(torch.from_numpy(ims[i_cam]).permute(2, 0, 1).to(self.dtype))
                cam_Ks_t[i_cam].append(K_scaled)
                cam_Tce_t[i_cam].append(T_cam_from_ego)
                cam_depths_t[i_cam].append(torch.from_numpy(d).to(self.dtype))
                cam_valids_t[i_cam].append(torch.from_numpy((d > 0.0).astype(np.uint8)))

            # ---- Target BEV mask (if you use box supervision on BEV) ----
            lidar_sample_token = self.nusc.get("sample_data", lidar_key)["sample_token"]
            ann = _ann_for_instance_at_sample(self.nusc, lidar_sample_token, inst_tok)
            m_box = bev_box_mask_from_ann(self, ann, lidar_key)                  # (H_bev, W_bev) uint8
            bev_mask_list.append(torch.from_numpy(m_box)[None, ...])

        # ---- Stack time-major camera tensors ----
        cam_imgs            = torch.stack([torch.stack(cam_imgs_t[i],   dim=0) for i in range(C)], dim=1)  # (T,C,3,H,cw)
        cam_depths          = torch.stack([torch.stack(cam_depths_t[i], dim=0) for i in range(C)], dim=1)  # (T,C,H,cw)
        cam_depth_valid     = torch.stack([torch.stack(cam_valids_t[i], dim=0) for i in range(C)], dim=1)  # (T,C,H,cw)
        cam_K_scaled        = torch.from_numpy(np.stack([np.stack(cam_Ks_t[i], axis=0) for i in range(C)], axis=1)).to(self.dtype)  # (T,C,3,3)
        cam_T_cam_from_ego  = torch.from_numpy(np.stack([np.stack(cam_Tce_t[i], axis=0) for i in range(C)], axis=1)).to(self.dtype) # (T,C,4,4)

        # ---- Time-major pillar tensors with per-sequence padding ----
        T_list = len(pillar_feats_t)
        maxPill = max((p.shape[0] for p in pillar_feats_t), default=0)
        feat_dim = pillar_feats_t[0].shape[-1] if maxPill > 0 else (10 if self.pp_include_dt else 9)
        maxPts = self.pp_max_points

        pillar_features = torch.zeros((T_list, maxPill, maxPts, feat_dim), dtype=self.dtype)
        pillar_coords   = torch.zeros((T_list, maxPill, 2), dtype=torch.int32)
        pillar_npoints  = torch.zeros((T_list, maxPill), dtype=torch.int32)

        for t in range(T_list):
            P_t = pillar_feats_t[t].shape[0]
            if P_t == 0:
                continue
            pillar_features[t, :P_t] = torch.from_numpy(pillar_feats_t[t]).to(self.dtype)
            pillar_coords[t,   :P_t] = torch.from_numpy(pillar_coords_t[t]).to(torch.int32)
            pillar_npoints[t,  :P_t] = torch.from_numpy(pillar_npts_t[t]).to(torch.int32)

        bev_target_mask = torch.stack(bev_mask_list, dim=0).to(torch.uint8)      # (T,1,H_bev,W_bev)

        # Grid size (H, W) for BEV scatter derived from PP voxel size
        grid_W = int(np.floor((self.bev_x_max - self.bev_x_min) / self.pp_voxel_size[0]))
        grid_H = int(np.floor((self.bev_y_max - self.bev_y_min) / self.pp_voxel_size[1]))

        return {
            # Cameras
            "cam_imgs":            cam_imgs,
            "cam_K_scaled":        cam_K_scaled,
            "cam_T_cam_from_ego":  cam_T_cam_from_ego,
            "cam_depths":          cam_depths,
            "cam_depth_valid":     cam_depth_valid,

            # PointPillars inputs (time-major)
            "pillar_features":     pillar_features,      # (T, P, M, C_feat) -> P≤min(H×W, max_pillars)
            "pillar_coords":       pillar_coords,        # (T, P, 2) [iy, ix]
            "pillar_num_points":   pillar_npoints,       # (T, P)

            # PP meta for scatter & bounds
            "pillar_meta": {
                "x_min": self.bev_x_min, "x_max": self.bev_x_max,
                "y_min": self.bev_y_min, "y_max": self.bev_y_max,
                "z_min": self.pp_z_bounds[0], "z_max": self.pp_z_bounds[1],
                "voxel_size": self.pp_voxel_size,        # (vx, vy, vz[unused])
                "max_points_per_pillar": self.pp_max_points,
                "max_pillars": self.pp_max_pillars,
                "include_dt": self.pp_include_dt,
                "grid_size_xy": (grid_H, grid_W)         # (H, W) for BEV tensor
            },

            # Supervision
            "bev_target_mask":     bev_target_mask,      # (T,1,H_bev,W_bev)
            "init_labels":         [1],                  # static single label for single target
            "traj":                traj,                 # (T_out,2)
            "last_pos":            last_pos,             # (2,)

            # Misc meta
            "meta": {
                "cams_order": self.trip,
                "bev": {
                    "x_min": self.bev_x_min, "x_max": self.bev_x_max,
                    "y_min": self.bev_y_min, "y_max": self.bev_y_max,
                    "H": self.H_bev, "W": self.W_bev,
                    "res_x": self.res_x, "res_y": self.res_y,
                    "ref_frame": "ego@t"
                },
            },
        }
