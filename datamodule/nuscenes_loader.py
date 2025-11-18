from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from trainer.utils import open_config
from data.configs.filenames import TRAIN_CONFIG, PP_CONFIG
from .image_utils import load_resize_arr
from .mask_utils import bev_box_mask_from_ann
from .lidar_utils import (
    T_cam_from_lidar_4x4,
    T_sensor_from_ego_4x4,
    transform_points,
    scale_K,
    rasterize_depth_xyz_cam,
    lidar_points_ego,
)
from .pillars_utils import aggregate_sweeps_to_anchor


def _ann_for_instance_at_sample(
    nusc: NuScenes, sample_token: str, instance_token: str
) -> Optional[dict]:
    # Locate this instance’s annotation within the sample (None if not present)
    s = nusc.get("sample", sample_token)
    for ann_tok in s["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        if ann["instance_token"] == instance_token:
            return ann
    return None


class NuScenesLoader(Dataset):
    def __init__(
        self,
        nusc: NuScenes,
        rows: List[Dict[str, Any]],
        dtype: torch.dtype = torch.float32,
    ):
        tr_cfg = open_config(TRAIN_CONFIG)  # Central config (sizes, bounds, discretization)
        pp_cfg = open_config(PP_CONFIG)

        self.nusc = nusc
        self.rows = rows
        self.dtype = dtype
        self.data_root = Path(self.nusc.dataroot).resolve()

        #Legacy flag for normalization or not
        self.normalize = True

        # Camera target sizes and ordering
        self.H = int(tr_cfg.get("H", 400))
        self.cw = int(tr_cfg.get("cw", 320))
        self.trip = tr_cfg.get("trip")  # Ordered camera names used consistently downstream

         # === Extract from PointPillars config === REWORKKKK
    
        # Get voxel encoder settings
        voxel_encoder = pp_cfg.get("pts_voxel_encoder", {})
        
        # Voxel size: [vx, vy, vz]
        self.pp_voxel_size = tuple(voxel_encoder.get("voxel_size", [0.2, 0.2, 4.0]))
        
        # Point cloud range: [x_min, y_min, z_min, x_max, y_max, z_max]
        pc_range = voxel_encoder.get("point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        self.bev_x_min, self.bev_y_min = float(pc_range[0]), float(pc_range[1])
        self.bev_x_max, self.bev_y_max = float(pc_range[3]), float(pc_range[4])
        
        # Z bounds for clipping
        self.pp_z_bounds = (float(pc_range[2]), float(pc_range[5]))
        
        # Max points per timestep
        self.pp_max_in_points = int(pp_cfg.get("max_num_points", 10)) * 1000  # Convert to actual point count
        # Or use a separate config value:
        # self.pp_max_in_points = int(tr_cfg.get("pp_max_in_points", 120000))

        # === Derive BEV grid shape ===
        vx, vy, _ = self.pp_voxel_size
        self.H_bev = 256
        self.W_bev = 256 
        self.res_y = vy
        self.res_x = vx

        # Cache for LiDAR
        self._lidar_xyz_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)  # Number of sequences/samples

    def resolve_path(self, p: str) -> str:
        # Make relative paths from the index absolute under dataroot
        q = Path(p)
        return str(q if q.is_absolute() else (self.data_root / q).resolve())

    def _load_lidar_xyz(self, sd_token: str) -> np.ndarray:
        # Read/cached LiDAR points (Nx3, meters) for a given sample_data token
        if sd_token in self._lidar_xyz_cache:
            return self._lidar_xyz_cache[sd_token]
        xyz = (
            LidarPointCloud.from_file(self.nusc.get_sample_data_path(sd_token))
            .points[:3, :]
            .T.astype(np.float32)
        )
        self._lidar_xyz_cache[sd_token] = xyz
        return xyz

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        # Establish camera order indices and time length
        cams_all: List[str] = row["cam_set"]
        cam_idx = [cams_all.index(c) for c in self.trip]  # map configured names → positions
        C = len(self.trip)
        T_in = len(row["obs_cam_img_grid"])              # number of timesteps

        # Per-camera temporal accumulators (time-major lists)
        cam_imgs_t:   List[List[torch.Tensor]] = [[] for _ in range(C)]
        cam_Ks_t:     List[List[np.ndarray]]   = [[] for _ in range(C)]
        cam_Tce_t:    List[List[np.ndarray]]   = [[] for _ in range(C)]
        cam_depths_t: List[List[torch.Tensor]] = [[] for _ in range(C)]
        cam_valids_t: List[List[torch.Tensor]] = [[] for _ in range(C)]

        # LiDAR points per timestep (ego frame)
        pts_list: List[torch.Tensor] = []

        # Supervision targets
        traj = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)  # (T_out, 2)
        last_pos = torch.tensor(row["target"]["last_xy"], dtype=self.dtype)  # (2,)
        inst_tok = row["target"]["agent_id"]  # Target instance token
        bev_mask_list: List[torch.Tensor] = []

        for t in range(T_in):
            # Resolve camera image paths for this timestep
            img_paths_rel = [row["obs_cam_img_grid"][t][cam_idx[i]] for i in range(C)]
            img_paths = [self.resolve_path(p) for p in img_paths_rel]

            ims, orig_hw = [], []
            for p in img_paths:
                im, (ow, oh) = load_resize_arr(self, p, self.cw, self.H)  # resize with preserved aspect
                ims.append(im)                                            # H×cw×3 ndarray, normalized per your helper
                orig_hw.append((oh, ow))                                  # needed to rescale intrinsics

            # LiDAR sweep tokens aligned to an anchor (keyframe) at this t
            lidar_sweeps: List[str] = row["lidar"]["sd_tokens"][t]
            lidar_key: str = row["lidar"].get("keyframe_sd_tokens", [lidar_sweeps[0]])[t]
            dt_list: List[float] = row["lidar"].get("dt_s", [[0.0] * len(lidar_sweeps)])[t]

            # Camera sample_data tokens and original intrinsics
            sd_cams = [row["cams"][self.trip[i]]["sd_tokens"][t] for i in range(C)]
            Ks_orig = [
                np.asarray(row["cams"][self.trip[i]]["intrinsics"], dtype=np.float32)
                for i in range(C)
            ]

            # Aggregate all sweeps into ego@keyframe with optional Δt as 5th channel
            pts_xyzit = aggregate_sweeps_to_anchor(
                self.nusc,
                lidar_sweeps,
                lidar_key,
                include_dt=True,                              # keeps Δt; downstream expects 5D points
                get_points_ego=lambda sd: lidar_points_ego(self, sd),
                dt_s=dt_list,
            )

            # Spatial clipping to BEV ROI and vertical bounds
            x_min, x_max = self.bev_x_min, self.bev_x_max
            y_min, y_max = self.bev_y_min, self.bev_y_max
            z_min, z_max = self.pp_z_bounds
            m = (
                (pts_xyzit[:, 0] >= x_min) & (pts_xyzit[:, 0] <= x_max) &
                (pts_xyzit[:, 1] >= y_min) & (pts_xyzit[:, 1] <= y_max) &
                (pts_xyzit[:, 2] >= z_min) & (pts_xyzit[:, 2] <= z_max)
            )
            pts_xyzit = pts_xyzit[m]

            # Hard cap for stability (random downsample if too dense)
            if pts_xyzit.shape[0] > self.pp_max_in_points:
                idxs = np.random.choice(
                    pts_xyzit.shape[0], self.pp_max_in_points, replace=False
                )
                pts_xyzit = pts_xyzit[idxs]

            # Store ego-frame points (x,y,z,intensity,Δt) as float tensor
            pts_list.append(torch.from_numpy(pts_xyzit).to(self.dtype))

            # Per-camera depth maps from union of sweeps projected into each camera
            for i_cam in range(C):
                (oh, ow) = orig_hw[i_cam]
                K_scaled = scale_K(Ks_orig[i_cam], (oh, ow), (self.H, self.cw))     # adjust intrinsics to resized image
                T_cam_from_ego = T_sensor_from_ego_4x4(self.nusc, sd_cams[i_cam])   # camera←ego at this time

                Xc_all = []
                for sdl in lidar_sweeps:
                    xyz_lidar = self._load_lidar_xyz(sdl)                           # cached raw sweep points
                    T_cl = T_cam_from_lidar_4x4(self.nusc, sd_cams[i_cam], sdl)     # camera←lidar transform
                    Xc_all.append(transform_points(T_cl, xyz_lidar))                # into camera frame
                Xc_cat = np.vstack(Xc_all) if Xc_all else np.zeros((0, 3), dtype=np.float32)

                d = rasterize_depth_xyz_cam(Xc_cat, K_scaled, (self.H, self.cw))    # z-buffer depth (H×cw)

                # Collect time-major tensors per camera
                cam_imgs_t[i_cam].append(torch.from_numpy(ims[i_cam]).permute(2, 0, 1).to(self.dtype))
                cam_Ks_t[i_cam].append(K_scaled)
                cam_Tce_t[i_cam].append(T_cam_from_ego)
                cam_depths_t[i_cam].append(torch.from_numpy(d).to(self.dtype))
                cam_valids_t[i_cam].append(torch.from_numpy((d > 0.0).astype(np.uint8)))

            # BEV supervision mask for the target instance at this time
            lidar_sample_token = self.nusc.get("sample_data", lidar_key)["sample_token"]
            ann = _ann_for_instance_at_sample(self.nusc, lidar_sample_token, inst_tok)
            m_box = bev_box_mask_from_ann(self, ann, lidar_key)                    # (H_bev, W_bev) uint8
            bev_mask_list.append(torch.from_numpy(m_box)[None, ...])               # add channel dim

        # Stack time-major camera tensors to shapes: (T, C, …)
        cam_imgs = torch.stack([torch.stack(cam_imgs_t[i], dim=0) for i in range(C)], dim=1)          # (T,C,3,H,cw)
        cam_depths = torch.stack([torch.stack(cam_depths_t[i], dim=0) for i in range(C)], dim=1)      # (T,C,H,cw)
        cam_depth_valid = torch.stack([torch.stack(cam_valids_t[i], dim=0) for i in range(C)], dim=1) # (T,C,H,cw)
        cam_K_scaled = torch.from_numpy(
            np.stack([np.stack(cam_Ks_t[i], axis=0) for i in range(C)], axis=1)
        ).to(self.dtype)                                                                              # (T,C,3,3)
        cam_T_cam_from_ego = torch.from_numpy(
            np.stack([np.stack(cam_Tce_t[i], axis=0) for i in range(C)], axis=1)
        ).to(self.dtype)                                                                              # (T,C,4,4)

        # Pack variable-length point sets into a (T, maxN, 5) tensor with zero-padding
        maxN = max((p.size(0) for p in pts_list), default=0)
        points = torch.zeros((T_in, maxN, 5), dtype=self.dtype)
        for t in range(T_in):
            n = pts_list[t].size(0)
            if n > 0:
                points[t, :n] = pts_list[t]

        # Time-major BEV mask tensor (T,1,H_bev,W_bev)
        bev_target_mask = torch.stack(bev_mask_list, dim=0).to(torch.uint8)

        # Grid size communicated to downstream encoders (for scatter/consistency)
        grid_W = int(np.floor((self.bev_x_max - self.bev_x_min) / self.pp_voxel_size[0]))
        grid_H = int(np.floor((self.bev_y_max - self.bev_y_min) / self.pp_voxel_size[1]))

        return {
            # Cameras (time-major)
            "cam_imgs": cam_imgs,                                   # (T,C,3,H,cw)
            "cam_K_scaled": cam_K_scaled,                           # (T,C,3,3)
            "cam_T_cam_from_ego": cam_T_cam_from_ego,               # (T,C,4,4)
            "cam_depths": cam_depths,                               # (T,C,H,cw)
            "cam_depth_valid": cam_depth_valid,                     # (T,C,H,cw, uint8)

            # LiDAR points for PointPillars (time-major)
            "points": points,                                       # (T,N,5) → [x,y,z,intensity,Δt]

            # BEV supervision
            "bev_target_mask": bev_target_mask,                     # (T,1,H_bev,W_bev)
            "init_labels": [1],                                     # single foreground class label

            # Trajectory targets
            "traj": traj,                                           # (T_out,2)
            "last_pos": last_pos,                                   # (2,)

            # Meta for alignment/consistency checks
            "meta": {
                "cams_order": self.trip,
                "bev": {
                    "x_min": self.bev_x_min,
                    "x_max": self.bev_x_max,
                    "y_min": self.bev_y_min,
                    "y_max": self.bev_y_max,
                    "H": self.H_bev,
                    "W": self.W_bev,
                    "res_x": self.res_x,
                    "res_y": self.res_y,
                    "ref_frame": "ego@t",
                },
                "pillar_meta": {
                    "voxel_size": self.pp_voxel_size,               # (vx,vy,vz) used by your PP encoder
                    "grid_size_xy": (grid_H, grid_W),               # (H,W) scatter grid for BEV tensors
                    "z_bounds": self.pp_z_bounds,                   # vertical clipping used here
                },
            },
        }
