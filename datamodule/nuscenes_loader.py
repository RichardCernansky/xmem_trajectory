from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

# ---- helpers you already have in your project ----
from trainer.utils import open_config
from data.configs.filenames import TRAIN_CONFIG
from .image_utils import load_resize_arr                          # (img_path, cw, H) -> (H,cw,3) + (orig_w,orig_h)
from .mask_utils import *
from .lidar_utils import (
    T_cam_from_lidar_4x4,                                         # (nusc, cam_sd_token, lidar_sd_token) -> 4x4
    T_sensor_from_ego_4x4,                                        # (nusc, sd_token) -> sensor<-ego (4x4)
    transform_points,                                             # (4x4, Nx3) -> Nx3
    scale_K,                                                      # (K, (orig_h,orig_w), (H,cw)) -> K_scaled
    rasterize_depth_xyz_cam,                                      # (Xc Nx3, K, (H,cw)) -> (H,cw) depth (m)
    lidar_points_ego,                                             # (loader, lidar_sd_token) -> (N,4) [x,y,z,intensity] in ego
    lidar_bev_from_points_fixed                                   # (loader, pts_ego) -> (4, H_bev, W_bev)
)

def _ann_for_instance_at_sample(nusc: NuScenes, sample_token: str, instance_token: str) -> Optional[dict]:
    """Return sample_annotation dict for this instance at this sample, else None."""
    s = nusc.get("sample", sample_token)
    for ann_tok in s["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        if ann["instance_token"] == instance_token:
            return ann
    return None

class NuScenesLoader(Dataset):
    """
    Minimal loader for BEV+XMem (no pano). Returns only the tensors the model needs:

      cam_imgs:            (T, C, 3, H, cw)           per-camera RGB (resized)
      cam_K_scaled:        (T, C, 3, 3)               intrinsics @ (H,cw)
      cam_T_cam_from_ego:  (T, C, 4, 4)               extrinsics sensor<-ego
      cam_depths:          (T, C, H, cw)              LiDAR z-buffer depth per cam (m)
      cam_depth_valid:     (T, C, H, cw) uint8        depth valid mask

      lidar_bev_raw:       (T, 4, H_bev, W_bev)       LiDAR BEV raw stats

      bev_target_center_px:(T, 2) int64               (iy, ix) center pixels in BEV
      bev_target_mask:     (T, 1, H_bev, W_bev) uint8 simple disk mask around center

      traj:                (T_out, 2)                 future XY (ego frame)
      last_pos:            (2,)                        last observed XY (ego)

      meta["bev"]: {x/y bounds, H/W, res_x/res_y, mapping}, meta["cams_order"]
    """
    def __init__(
        self,
        nusc: NuScenes,
        rows: List[Dict[str, Any]],
        dtype: torch.dtype = torch.float32,
    ):
        # pull sizes/ROI/camera order from TRAIN_CONFIG
        train_config = open_config(TRAIN_CONFIG)

        # core refs
        self.nusc = nusc
        self.rows = rows
        self.dtype = dtype

        self.data_root = Path(self.nusc.dataroot).resolve()

        self.normalize = True
        # image sizing (per-camera)
        self.H  = int(train_config.get("H", 400))
        self.cw = int(train_config.get("cw", 320))

        # camera triplet order
        self.trip = train_config.get("trip")

        # BEV metric ROI (meters) — shared by LiDAR and RGB splats
        bev_x_bounds = train_config.get("bev_x_bounds", [-5.0, 55.0])
        bev_y_bounds = train_config.get("bev_y_bounds", [-30.0, 30.0])

        # Fixed BEV grid (independent of image size)
        H_bev = int(train_config.get("H_bev", 150))
        W_bev = int(train_config.get("W_bev", 150))

        # BEV spec
        self.bev_x_min, self.bev_x_max = map(float, bev_x_bounds)
        self.bev_y_min, self.bev_y_max = map(float, bev_y_bounds)
        self.H_bev = H_bev
        self.W_bev = W_bev
        self.res_x = (self.bev_x_max - self.bev_x_min) / float(self.W_bev)  # m/col
        self.res_y = (self.bev_y_max - self.bev_y_min) / float(self.H_bev)  # m/row


    def __len__(self) -> int:
        return len(self.rows)

    def resolve_path(self, p: str) -> str:
            q = Path(p)
            return str(q if q.is_absolute() else (self.data_root / q).resolve())


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        cams_all: List[str] = row["cam_set"]
        cam_idx = [cams_all.index(c) for c in self.trip]
        C = len(self.trip)

        T_in = len(row["obs_cam_img_grid"])

        # Per-camera containers (time-major)
        cam_imgs_t:   List[List[torch.Tensor]] = [[] for _ in range(C)]
        cam_Ks_t:     List[List[np.ndarray]]   = [[] for _ in range(C)]
        cam_Tce_t:    List[List[np.ndarray]]   = [[] for _ in range(C)]
        cam_depths_t: List[List[torch.Tensor]] = [[] for _ in range(C)]
        cam_valids_t: List[List[torch.Tensor]] = [[] for _ in range(C)]

        # LiDAR BEV & target markers
        lidar_bev_raw_list: List[torch.Tensor] = []
        bev_mask_list:   List[torch.Tensor]    = []

        # Supervision
        # (traj and last_pos are already in ego frame at anchor time per your index)
        traj     = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)
        last_pos = torch.tensor(row["target"]["last_xy"],   dtype=self.dtype)

        inst_tok = row["target"]["agent_id"]

        for t in range(T_in):

            # -------- per-camera RGB (resized), intrinsics, extrinsics, depth --------
            img_paths_rel = [row["obs_cam_img_grid"][t][cam_idx[i]] for i in range(C)]
            img_paths = [self.resolve_path(p) for p in img_paths_rel]
            ims = []
            orig_hw = []
            for p in img_paths:
                im, (ow, oh) = load_resize_arr(self, p, self.cw, self.H)      # (H,cw,3) + (orig_w,orig_h)
                ims.append(im)
                orig_hw.append((oh, ow))                                      # store (orig_h,orig_w)

            # tokens
            lidar_sd = row["lidar"]["sd_tokens"][t]
            sd_cams = [row["cams"][self.trip[i]]["sd_tokens"][t] for i in range(C)]
            Ks_orig = [np.asarray(row["cams"][self.trip[i]]["intrinsics"], dtype=np.float32) for i in range(C)]

            # LiDAR→ego points & BEV raw
            pts_ego = lidar_points_ego(self, lidar_sd)                        # (N,4)
            bev_raw = lidar_bev_from_points_fixed(self, pts_ego)              # (4,H_bev,W_bev)
            lidar_bev_raw_list.append(torch.from_numpy(bev_raw))

            # depth needs LiDAR-frame xyz
            pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_sd)).points
            xyz_lidar = pc[:3, :].T.astype(np.float32)

            for i in range(C):
                (oh, ow) = orig_hw[i]
                K_scaled = scale_K(Ks_orig[i], (oh, ow), (self.H, self.cw))    # (3,3)
                T_cam_from_ego = T_sensor_from_ego_4x4(self.nusc, sd_cams[i])  # (4,4)
                T_cam_from_lidar = T_cam_from_lidar_4x4(self.nusc, sd_cams[i], lidar_sd)

                # depth via LiDAR→camera projection
                Xc = transform_points(T_cam_from_lidar, xyz_lidar)             # (N,3)
                d = rasterize_depth_xyz_cam(Xc, K_scaled, (self.H, self.cw))   # (H,cw), meters; 0=no hit

                cam_imgs_t[i].append(torch.from_numpy(ims[i]).permute(2,0,1).to(self.dtype))
                cam_Ks_t[i].append(K_scaled)
                cam_Tce_t[i].append(T_cam_from_ego)
                cam_depths_t[i].append(torch.from_numpy(d).to(self.dtype))
                cam_valids_t[i].append(torch.from_numpy((d > 0.0).astype(np.uint8)))

            # -------- target BEV mask from oriented box --------
            lidar_sample_token = self.nusc.get("sample_data", lidar_sd)["sample_token"]
            ann = _ann_for_instance_at_sample(self.nusc, lidar_sample_token, inst_tok)

            poly_px = ann_to_bev_polygon_pixels(self, ann, lidar_sd)  # (4,2) int32 (x=col, y=row)
            m_box   = bev_box_mask_from_ann(self, ann, lidar_sd)      # (H_bev, W_bev) uint8

            # store mask
            bev_mask_list.append(torch.from_numpy(m_box)[None, ...])

        # stack per-camera over time -> (T,C,...)
        cam_imgs            = torch.stack([torch.stack(cam_imgs_t[i],   dim=0) for i in range(C)], dim=1)  # (T,C,3,H,cw)
        cam_depths          = torch.stack([torch.stack(cam_depths_t[i], dim=0) for i in range(C)], dim=1)  # (T,C,H,cw)
        cam_depth_valid     = torch.stack([torch.stack(cam_valids_t[i], dim=0) for i in range(C)], dim=1)  # (T,C,H,cw)
        cam_K_scaled        = torch.from_numpy(np.stack([np.stack(cam_Ks_t[i], axis=0) for i in range(C)], axis=1)).to(self.dtype)   # (T,C,3,3)
        cam_T_cam_from_ego  = torch.from_numpy(np.stack([np.stack(cam_Tce_t[i], axis=0) for i in range(C)], axis=1)).to(self.dtype)  # (T,C,4,4)

        lidar_bev_raw = torch.stack(lidar_bev_raw_list, dim=0)            # (T,4,H_bev,W_bev)
        bev_target_mask = torch.stack(bev_mask_list, dim=0).to(torch.uint8)  # (T,1,H_bev,W_bev)

        return {
            # Per-camera inputs for Lift/Splat
            "cam_imgs":            cam_imgs,             # (T,C,3,H,cw)
            "cam_K_scaled":        cam_K_scaled,         # (T,C,3,3)
            "cam_T_cam_from_ego":  cam_T_cam_from_ego,   # (T,C,4,4)
            "cam_depths":          cam_depths,           # (T,C,H,cw)
            "cam_depth_valid":     cam_depth_valid,      # (T,C,H,cw) uint8

            # LiDAR BEV raw (fixed grid)
            "lidar_bev_raw":       lidar_bev_raw,        # (T,4,H_bev,W_bev)

            "bev_target_mask":      bev_target_mask,       # (T,1,H_bev,W_bev) uint8 (disk)
            "init_labels":   [1],
            # Supervision
            "traj":          traj,  # (T_out,2)
            "last_pos":      last_pos,  # (2,)

            # Meta needed for consistent BEV mapping
            "meta": {
                "cams_order": self.trip,
                "bev": {
                    "x_min": self.bev_x_min, "x_max": self.bev_x_max,
                    "y_min": self.bev_y_min, "y_max": self.bev_y_max,
                    "H": self.H_bev, "W": self.W_bev,
                    "res_x": self.res_x, "res_y": self.res_y,
                    "mapping": "y_min→top rows, y_max→bottom rows; x_min→left cols, x_max→right cols",
                    "ref_frame": "ego@t"
                },
            },
        }
