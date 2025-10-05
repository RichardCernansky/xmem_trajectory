from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from .image_utils import compute_physical_overlaps, load_resize_arr, compose_three
from .mask_utils import mask_from_box, compose_masks

# LiDAR helpers
from .lidar_utils import (
    T_cam_from_lidar_4x4,  # builds cam←lidar 4x4 transform at the correct timestamps
    scale_K,               # scales intrinsics to (H, cw)
    transform_points,      # applies 4x4 to Nx3 points
    rasterize_depth_xyz_cam,  # projects + z-buffers into a (H,cw) depth map
    compose_three_depth    # stitches three depth crops into one pano using "near wins"
)

class NuScenesLoader(Dataset):
    def __init__(
        self,
        nusc: NuScenes,
        rows: List[Dict[str, Any]],
        out_size: Tuple[int, int]=(384, 640),
        img_normalize: bool=True,
        dtype: torch.dtype=torch.float32,
        crop_w: int=320,
        pano_triplet: Optional[List[str]]=None,
        min_overlap_ratio: float=0.15,
        max_overlap_ratio: float=0.6,
    ):
        self.nusc = nusc
        self.rows = rows
        self.H = int(out_size[0])
        self.normalize = img_normalize
        self.dtype = dtype
        self.cw = int(crop_w)
        self.pano_triplet = pano_triplet or ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
        self.min_or = float(min_overlap_ratio)
        self.max_or = float(max_overlap_ratio)

        self.trip = self.pano_triplet
        # Compute pano overlaps once (px) for L–F and F–R
        self.ov_lf_px, self.ov_fr_px = compute_physical_overlaps(self, self.rows[0], self.trip, self.cw)
        self.ov_lf_px = int(np.clip(self.ov_lf_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.ov_fr_px = int(np.clip(self.ov_fr_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.W = 3*self.cw - (self.ov_lf_px + self.ov_fr_px)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        obs_paths_grid: List[List[str]] = row["obs_cam_img_grid"]
        cams_all: List[str] = row["cam_set"]

        # Use the 3-camera triplet provided
        trip = cams_all
        idxs = [cams_all.index(c) for c in trip]

        T_in = len(obs_paths_grid)
        frames_t: List[torch.Tensor] = []
        depths_t: List[torch.Tensor] = []

        inst_tok: str = row["target"]["agent_id"]
        m0L = m0F = m0R = None
        owL = ohL = owF = ohF = owR = ohR = None

        for t in range(T_in):
            # === RGB: load + resize + stitch (your existing part) ===
            pL = obs_paths_grid[t][idxs[0]]
            pF = obs_paths_grid[t][idxs[1]]
            pR = obs_paths_grid[t][idxs[2]]

            imL, (owL, ohL) = load_resize_arr(self, pL, self.cw, self.H)  # returns (image, (orig_w, orig_h))
            imF, (owF, ohF) = load_resize_arr(self, pF, self.cw, self.H)
            imR, (owR, ohR) = load_resize_arr(self, pR, self.cw, self.H)

            comp = compose_three(self, imL, imF, imR)
            frames_t.append(torch.from_numpy(comp).permute(2, 0, 1).to(self.dtype))

            # === LiDAR: depth for this timestep t (on-the-fly) ===

            # 1) Find the LiDAR sample_data token for this sample (paired to current t)
            samp = self.nusc.get("sample", row["obs_sample_tokens"][t])
            lidar_sd = samp["data"]["LIDAR_TOP"]  # token of the LiDAR sweep at this sample

            # 2) Load point cloud for this LiDAR sweep (x,y,z,intensity in meters)
            pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_sd)).points
            xyz = pc[:3, :].T.astype(np.float32)  # shape (N,3), discard intensity

            # 3) Per-camera sample_data tokens at this timestep (for cam-specific transforms)
            sdLt = row["cams"][trip[0]]["sd_tokens"][t]
            sdFt = row["cams"][trip[1]]["sd_tokens"][t]
            sdRt = row["cams"][trip[2]]["sd_tokens"][t]

            # 4) Build cam←lidar transforms using calibration + poses at their own timestamps
            TL = T_cam_from_lidar_4x4(self.nusc, sdLt, lidar_sd)
            TF = T_cam_from_lidar_4x4(self.nusc, sdFt, lidar_sd)
            TR = T_cam_from_lidar_4x4(self.nusc, sdRt, lidar_sd)

            # 5) Transform LiDAR points into each camera coordinate frame
            XcL = transform_points(TL, xyz)  # (N,3) in Left camera frame
            XcF = transform_points(TF, xyz)  # (N,3) in Front camera frame
            XcR = transform_points(TR, xyz)  # (N,3) in Right camera frame

            # 6) Fetch original intrinsics for each camera
            KL = np.asarray(row["cams"][trip[0]]["intrinsics"], dtype=np.float32)
            KF = np.asarray(row["cams"][trip[1]]["intrinsics"], dtype=np.float32)
            KR = np.asarray(row["cams"][trip[2]]["intrinsics"], dtype=np.float32)

            # 7) Scale intrinsics to the resized crop size (H,cw)
            #    Note: load_resize_arr returns (orig_w, orig_h), but scale_K expects (orig_h, orig_w)
            KLs = scale_K(KL, (ohL, owL), (self.H, self.cw))
            KFs = scale_K(KF, (ohF, owF), (self.H, self.cw))
            KRs = scale_K(KR, (ohR, owR), (self.H, self.cw))

            # 8) Project each camera's points and z-buffer into a per-camera depth crop (H,cw)
            dL = rasterize_depth_xyz_cam(XcL, KLs, (self.H, self.cw))  # meters, 0.0 = no hit
            dF = rasterize_depth_xyz_cam(XcF, KFs, (self.H, self.cw))
            dR = rasterize_depth_xyz_cam(XcR, KRs, (self.H, self.cw))

            # 9) Stitch three per-camera depth crops into one panoramic depth (H,W)
            pano_depth = compose_three_depth(dL, dF, dR, self.cw, self.ov_lf_px, self.ov_fr_px)

            # 10) Store as (1,H,W) tensor for this timestep
            depths_t.append(torch.from_numpy(pano_depth[None, ...]).to(self.dtype))

            # === Initial mask on t==0 (your existing part) ===
            if t == 0:
                sdL0 = row["cams"][trip[0]]["sd_tokens"][0]
                sdF0 = row["cams"][trip[1]]["sd_tokens"][0]
                sdR0 = row["cams"][trip[2]]["sd_tokens"][0]
                KL0 = KL; KF0 = KF; KR0 = KR
                m0L = mask_from_box(self, self.nusc, sdL0, inst_tok, KL0, (self.H, self.cw), (owL, ohL))
                m0F = mask_from_box(self, self.nusc, sdF0, inst_tok, KF0, (self.H, self.cw), (owF, ohF))
                m0R = mask_from_box(self, self.nusc, sdR0, inst_tok, KR0, (self.H, self.cw), (owR, ohR))

        # Stack over time: RGB (T,3,H,W), Depth (T,1,H,W)
        frames = torch.stack(frames_t, dim=0)
        depths = torch.stack(depths_t, dim=0)

        # Build initial panorama mask
        if m0L is None:
            init_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        else:
            init_mask = compose_masks(self, m0L, m0F, m0R)
        init_mask_t = torch.from_numpy(init_mask[None, ...]).to(torch.uint8)

        # Trajectory supervision
        traj = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)
        last_pos = torch.tensor(row["target"]["last_xy"], dtype=self.dtype)

        # Final sample
        return {
            "frames": frames,          # (T_in, 3, H, W) RGB pano
            "depths": depths,          # (T_in, 1, H, W) depth pano (meters; 0 = no hit)
            "traj": traj,              # (T_future, 2)
            "last_pos": last_pos,      # (2,)
            "init_masks": init_mask_t, # (1, H, W) uint8
            "init_labels": [1],
            "meta": {
                "scene_name": row["scene_name"],
                "cams": trip,
                "start_sample_token": row["start_sample_token"],
                "pano_w": self.W,
                "H": self.H,
                "overlap_lf": self.ov_lf_px,
                "overlap_fr": self.ov_fr_px,
                "crop_w": self.cw,
            },
        }
