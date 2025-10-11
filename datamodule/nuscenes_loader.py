from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# helpers you already have
from .image_utils import compute_physical_overlaps, load_resize_arr, compose_three
from .mask_utils import mask_from_box, compose_masks
from .lidar_utils import (
    T_cam_from_lidar_4x4,
    T_sensor_from_ego_4x4,     # sensor←ego (invert → ego←sensor)
    transform_points,
    scale_K,
    rasterize_depth_xyz_cam,
    compose_three_depth,
    invdepth_valid_from_depth
)

class NuScenesLoader(Dataset):
    """
    Outputs:
      - frames:        (T, 3, H_pano, W_pano)
      - depths:        (T, 1, H_pano, W_pano)
      - depth_extras:  (T, 2, H_pano, W_pano)  # [inv_depth_0..1, valid]
      - init_masks:    (1, H_pano, W_pano)     # uint8 pano mask @t=0
      - lidar_bev_raw: (T, 4, H_raw, W_raw)    # [log1p(count), mean_z, max_z, mean_intensity]
      - traj:          (T_out, 2)
      - last_pos:      (2,)
      - meta:          dict with pano + BEV specs
    """
    def __init__(
        self,
        nusc: NuScenes,
        rows: List[Dict[str, Any]],
        H_pano: int = 400,              # pano height
        crop_w: int = 320,
        pano_triplet: Optional[List[str]] = None,
        min_overlap_ratio: float = 0.15,
        max_overlap_ratio: float = 0.60,
        dtype: torch.dtype = torch.float32,

        # --- BEV metric ROI (meters) ---
        bev_x_bounds: Tuple[float, float] = (-5.0, 55.0),   # x forward: 60m span
        bev_y_bounds: Tuple[float, float] = (-30.0, 30.0),  # y left/right: 60m span

        # --- FIXED BEV grid shape (H_raw, W_raw) ---
        H_raw: int = 150,
        W_raw: int = 150,
    ):
        self.nusc = nusc
        self.rows = rows
        self.H = int(H_pano)    # pano H
        self.cw = int(crop_w)
        self.trip = pano_triplet or ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
        self.min_or = float(min_overlap_ratio)
        self.max_or = float(max_overlap_ratio)
        self.dtype = dtype

        # Pano overlaps & pano width
        self.ov_lf_px, self.ov_fr_px = compute_physical_overlaps(self, self.rows[0], self.trip, self.cw)
        self.ov_lf_px = int(np.clip(self.ov_lf_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.ov_fr_px = int(np.clip(self.ov_fr_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.W = 3*self.cw - (self.ov_lf_px + self.ov_fr_px)   # pano W (independent of BEV)

        # BEV ROI + fixed grid (H_raw, W_raw) → infer meters-per-pixel (anisotropic)
        self.bev_x_min, self.bev_x_max = map(float, bev_x_bounds)
        self.bev_y_min, self.bev_y_max = map(float, bev_y_bounds)
        self.H_raw = int(H_raw)
        self.W_raw = int(W_raw)
        self.res_x = (self.bev_x_max - self.bev_x_min) / float(self.W_raw)   # m per column
        self.res_y = (self.bev_y_max - self.bev_y_min) / float(self.H_raw)   # m per row

    def __len__(self) -> int:
        return len(self.rows)

    def _lidar_points_ego(self, lidar_sd_token: str) -> np.ndarray:
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

    def _lidar_bev_from_points_fixed(self, pts_ego: np.ndarray) -> np.ndarray:
        """
        Aggregate ego-frame points into a FIXED BEV grid: (H_raw, W_raw),
        channels = [log1p(count), mean_z, max_z, mean_intensity].
        Mapping: y_min -> top rows, y_max -> bottom rows (image rows increase downward).
        """
        C = 4
        bev = np.zeros((C, self.H_raw, self.W_raw), dtype=np.float32)
        if pts_ego.size == 0:
            return bev

        x = pts_ego[:, 0]; y = pts_ego[:, 1]; z = pts_ego[:, 2]; inten = pts_ego[:, 3]

        # ROI filter
        m = (x >= self.bev_x_min) & (x < self.bev_x_max) & (y >= self.bev_y_min) & (y < self.bev_y_max)
        if not np.any(m):
            return bev
        x = x[m]; y = y[m]; z = z[m]; inten = inten[m]

        # Discretize with inferred meters-per-pixel
        ix = np.floor((x - self.bev_x_min) / self.res_x).astype(np.int64)  # [0..W_raw-1]
        iy = np.floor((y - self.bev_y_min) / self.res_y).astype(np.int64)  # [0..H_raw-1]
        np.clip(ix, 0, self.W_raw - 1, out=ix)
        np.clip(iy, 0, self.H_raw - 1, out=iy)
        lin = iy * self.W_raw + ix  # flattened indices

        HW = self.H_raw * self.W_raw
        count = np.bincount(lin, minlength=HW).astype(np.float32)
        sum_z = np.bincount(lin, weights=z, minlength=HW).astype(np.float32)
        sum_i = np.bincount(lin, weights=inten, minlength=HW).astype(np.float32)
        max_z = np.full(HW, -np.inf, dtype=np.float32)
        np.maximum.at(max_z, lin, z)

        eps = 1e-6
        mean_z = sum_z / (count + eps)
        mean_i = sum_i / (count + eps)

        # reshape & stabilize
        count = count.reshape(self.H_raw, self.W_raw)
        mean_z = mean_z.reshape(self.H_raw, self.W_raw)
        max_z  = max_z.reshape(self.H_raw, self.W_raw)
        mean_i = mean_i.reshape(self.H_raw, self.W_raw)

        mean_z = np.clip(mean_z, -3.0, 5.0)
        max_z  = np.clip(max_z,  -3.0, 5.0)
        mean_i = np.clip(mean_i,  0.0, 1.0)

        bev[0] = np.log1p(count)
        bev[1] = mean_z
        bev[2] = max_z
        bev[3] = mean_i
        return bev

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        cams_all: List[str] = row["cam_set"]
        trip = self.trip
        t_idxs = [cams_all.index(c) for c in trip]

        T_in = len(row["obs_cam_img_grid"])

        frames_t: List[torch.Tensor] = []
        depth_t:  List[torch.Tensor] = []
        extras_t: List[torch.Tensor] = []
        lidar_bev_raw_list: List[torch.Tensor] = []

        m0L = m0F = m0R = None

        for t in range(T_in):
            # --- RGB pano (stitched) ---
            pL = row["obs_cam_img_grid"][t][t_idxs[0]]
            pF = row["obs_cam_img_grid"][t][t_idxs[1]]
            pR = row["obs_cam_img_grid"][t][t_idxs[2]]

            imL, (owL, ohL) = load_resize_arr(self, pL, self.cw, self.H)
            imF, (owF, ohF) = load_resize_arr(self, pF, self.cw, self.H)
            imR, (owR, ohR) = load_resize_arr(self, pR, self.cw, self.H)
            pano_rgb = compose_three(self, imL, imF, imR, self.ov_lf_px, self.ov_fr_px)  # (H_pano,W_pano,3)
            frames_t.append(torch.from_numpy(pano_rgb).permute(2,0,1).to(self.dtype))

            # --- LiDAR ego points -> fixed-grid BEV raw ---
            lidar_sd = row["lidar"]["sd_tokens"][t]
            pts_ego = self._lidar_points_ego(lidar_sd)
            bev_raw = self._lidar_bev_from_points_fixed(pts_ego)  # (4, H_raw, W_raw)
            lidar_bev_raw_list.append(torch.from_numpy(bev_raw))

            # --- LiDAR→camera depth pano (for RGB lift/splat guidance) ---
            sdLt = row["cams"][trip[0]]["sd_tokens"][t]
            sdFt = row["cams"][trip[1]]["sd_tokens"][t]
            sdRt = row["cams"][trip[2]]["sd_tokens"][t]

            TL = T_cam_from_lidar_4x4(self.nusc, sdLt, lidar_sd)
            TF = T_cam_from_lidar_4x4(self.nusc, sdFt, lidar_sd)
            TR = T_cam_from_lidar_4x4(self.nusc, sdRt, lidar_sd)

            KL = np.asarray(row["cams"][trip[0]]["intrinsics"], dtype=np.float32)
            KF = np.asarray(row["cams"][trip[1]]["intrinsics"], dtype=np.float32)
            KR = np.asarray(row["cams"][trip[2]]["intrinsics"], dtype=np.float32)

            KLs = scale_K(KL, (ohL, owL), (self.H, self.cw))
            KFs = scale_K(KF, (ohF, owF), (self.H, self.cw))
            KRs = scale_K(KR, (ohR, owR), (self.H, self.cw))

            pc = LidarPointCloud.from_file(self.nusc.get_sample_data_path(lidar_sd)).points
            xyz_lidar = pc[:3, :].T.astype(np.float32)

            XcL = transform_points(TL, xyz_lidar)
            XcF = transform_points(TF, xyz_lidar)
            XcR = transform_points(TR, xyz_lidar)

            dL = rasterize_depth_xyz_cam(XcL, KLs, (self.H, self.cw))
            dF = rasterize_depth_xyz_cam(XcF, KFs, (self.H, self.cw))
            dR = rasterize_depth_xyz_cam(XcR, KRs, (self.H, self.cw))

            pano_depth = compose_three_depth(dL, dF, dR, self.cw, self.ov_lf_px, self.ov_fr_px)  # (H_pano, W_pano)
            depth_t.append(torch.from_numpy(pano_depth[None, ...]).to(self.dtype))

            inv01, valid = invdepth_valid_from_depth(pano_depth)
            extras = np.stack([inv01, valid], axis=0).astype(np.float32)
            extras_t.append(torch.from_numpy(extras))

            # --- Init pano mask @ t=0 ---
            if t == 0:
                m0L = mask_from_box(self, self.nusc, sdLt, row["target"]["agent_id"], KLs, (self.H, self.cw), (owL, ohL))
                m0F = mask_from_box(self, self.nusc, sdFt, row["target"]["agent_id"], KFs, (self.H, self.cw), (owF, ohF))
                m0R = mask_from_box(self, self.nusc, sdRt, row["target"]["agent_id"], KRs, (self.H, self.cw), (owR, ohR))

        # Stack time
        frames        = torch.stack(frames_t, dim=0)          # (T, 3, H_pano, W_pano)
        depths        = torch.stack(depth_t,  dim=0)          # (T, 1, H_pano, W_pano)
        depth_extras  = torch.stack(extras_t, dim=0)          # (T, 2, H_pano, W_pano)
        lidar_bev_raw = torch.stack(lidar_bev_raw_list, dim=0)  # (T, 4, H_raw, W_raw)

        # Init pano mask
        if m0L is None:
            init_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        else:
            init_mask = compose_masks(self, m0L, m0F, m0R, self.ov_lf_px, self.ov_fr_px)
        init_masks = torch.from_numpy(init_mask[None, ...]).to(torch.uint8)

        # Supervision
        traj     = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)
        last_pos = torch.tensor(row["target"]["last_xy"],   dtype=self.dtype)

        return {
            "frames":        frames,
            "depths":        depths,
            "depth_extras":  depth_extras,
            "init_masks":    init_masks,
            "init_labels":   [1],
            "lidar_bev_raw": lidar_bev_raw,   # (T, 4, 150, 150) by default
            "traj":          traj,
            "last_pos":      last_pos,
            "meta": {
                "scene_name": row["scene_name"],
                "start_sample_token": row["start_sample_token"],
                "pano": {
                    "H": self.H, "W": self.W,
                    "overlap_lf": self.ov_lf_px, "overlap_fr": self.ov_fr_px,
                    "crop_w": self.cw, "cams": self.trip
                },
                "bev": {
                    "x_min": self.bev_x_min, "x_max": self.bev_x_max,
                    "y_min": self.bev_y_min, "y_max": self.bev_y_max,
                    "H": self.H_raw, "W": self.W_raw,
                    "res_x": self.res_x, "res_y": self.res_y,
                    "mapping": "y_min→top rows, y_max→bottom rows; x_min→left cols, x_max→right cols",
                    "ref_frame": "ego@t"  # each timestep uses its own ego pose
                },
            },
        }
