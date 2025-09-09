# traj/datamodules.py
from typing import Any, Dict, List, Tuple, Optional

import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud
from traj.lidar_projection import lidar_to_2d_maps, adjust_intrinsics

# -------------------- image helpers --------------------
def _load_and_resize_rgb(path: str, hw: Tuple[int, int]) -> torch.Tensor:
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = hw
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    return t

# -------------------- mask helpers (t=0 only) --------------------
def _project_box_to_poly2d_safe(box, cam_K: np.ndarray, im_w: int, im_h: int) -> Optional[np.ndarray]:
    corners_3d = box.corners()  # (3, 8)
    z = corners_3d[2, :]
    valid = z > 1e-6
    if valid.sum() < 3:
        return None
    corners_3d = corners_3d[:, valid]
    pts = view_points(corners_3d, cam_K, normalize=True)[:2, :].T
    if not np.isfinite(pts).all():
        pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None
    pts[:, 0] = np.clip(pts[:, 0], 0, im_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, im_h - 1)
    pts_cv = np.ascontiguousarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_cv).reshape(-1, 2).astype(np.int32)
    if hull.shape[0] < 3:
        return None
    return hull

def masks_from_t0(
    nusc: NuScenes,
    cam_sd_token: str,
    resize_hw: Tuple[int, int],
    classes_prefix: Tuple[str, ...] = ("vehicle.car", "vehicle.truck", "vehicle.bus"),
    max_K: int = 4,
) -> Tuple[torch.Tensor, List[int]]:
    img_path, boxes, K = nusc.get_sample_data(cam_sd_token, box_vis_level=0)
    im_bgr = cv2.imread(img_path)
    if im_bgr is None:
        raise FileNotFoundError(img_path)
    im_h, im_w = im_bgr.shape[:2]
    H, W = resize_hw
    selected = []
    for b in boxes:
        if any(b.name.startswith(p) for p in classes_prefix):
            dist = float(np.linalg.norm(b.center))
            selected.append((dist, b))
    selected.sort(key=lambda x: x[0])
    selected = [b for _, b in selected[:max_K]]
    masks = []
    for b in selected:
        poly = _project_box_to_poly2d_safe(b, np.array(K, dtype=np.float32), im_w, im_h)
        m = np.zeros((im_h, im_w), dtype=np.uint8)
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(m, [poly], 1)
        if (im_h, im_w) != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(torch.from_numpy(m).float())
    if len(masks) == 0:
        M = torch.ones(1, H, W, dtype=torch.float32)
        labels = [1]
    else:
        M = torch.stack(masks, dim=0)
        labels = list(range(1, M.size(0) + 1))
    return M, labels

# -------------------- Dataset --------------------
class NuScenesSeqLoader(Dataset):
    """
    Expects index rows to include: img_paths, intrinsics, lidar_paths, T_cam_lidar,
    cam_sd_tokens, obs_sample_tokens, fut_sample_tokens, target{last_xy,future_xy}.
    """
    def __init__(
        self,
        index_path: str,
        resize: bool = False,
        resize_wh: Tuple[int, int] = (960, 540),      # (W, H)
        nusc: Optional[NuScenes] = None,              # if provided, we build t=0 masks
        classes_prefix: Tuple[str, ...] = ("vehicle.car", "vehicle.truck", "vehicle.bus"),
        max_K: int = 4,
        lidar_max_depth: float = 80.0,
        lidar_h_min: float = -2.0,
        lidar_h_max: float = 4.0,
        lidar_count_clip: float = 5.0,
    ):
        super().__init__()
        with open(index_path, "rb") as f:
            self.index: List[Dict[str, Any]] = pickle.load(f)

        self.resize = resize
        self.resize_wh = resize_wh
        self.nusc = nusc
        self.classes_prefix = classes_prefix
        self.max_K = max_K

        self.lidar_max_depth = lidar_max_depth
        self.lidar_h_min = lidar_h_min
        self.lidar_h_max = lidar_h_max
        self.lidar_count_clip = lidar_count_clip

        assert len(self.index) > 0, "Empty index file."
        req = {"img_paths","intrinsics","lidar_paths","T_cam_lidar","cam_sd_tokens","target"}
        missing = req - set(self.index[0].keys())
        if missing:
            raise KeyError(f"Index entries missing keys: {missing}")
        self.t_in  = len(self.index[0]["img_paths"])
        self.t_out = len(self.index[0]["target"]["future_xy"])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.index[idx]

        # determine output size
        im0 = cv2.imread(e["img_paths"][0])
        if im0 is None:
            raise FileNotFoundError(e["img_paths"][0])
        orig_h0, orig_w0 = im0.shape[:2]
        if self.resize:
            W, H = self.resize_wh
        else:
            W, H = orig_w0, orig_h0

        # RGB frames
        frames = torch.zeros(self.t_in, 3, H, W, dtype=torch.float32)
        for t, p in enumerate(e["img_paths"]):
            if self.resize:
                frames[t] = _load_and_resize_rgb(p, (H, W))
            else:
                # keep native; convert to tensor
                im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                timg = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
                frames[t] = timg

        # LiDAR maps
        lidar_maps = torch.zeros(self.t_in, 5, H, W, dtype=torch.float32)
        for t, (img_p, K_list, lidar_p, T_list) in enumerate(
            zip(e["img_paths"], e["intrinsics"], e["lidar_paths"], e["T_cam_lidar"])
        ):
            im_bgr = cv2.imread(img_p)
            if im_bgr is None:
                raise FileNotFoundError(img_p)
            im_h, im_w = im_bgr.shape[:2]
            K = np.asarray(K_list, dtype=np.float32)
            if self.resize:
                K = adjust_intrinsics(K, (im_w, im_h), (W, H))
            pc = LidarPointCloud.from_file(lidar_p).points.T  # [N,4] xyz,i
            T_cl = np.asarray(T_list, dtype=np.float32)
            maps = lidar_to_2d_maps(
                points_xyz_i=pc,
                K=K,
                T_cam_lidar=T_cl,
                W=W, H=H,
                max_depth=self.lidar_max_depth,
                h_min=self.lidar_h_min,
                h_max=self.lidar_h_max,
                count_clip=self.lidar_count_clip,
            )
            lidar_maps[t] = maps

        # masks at t=0 (optional)
        if self.nusc is not None:
            cam_sd_t0 = e["cam_sd_tokens"][0]
            init_masks, init_labels = masks_from_t0(
                nusc=self.nusc,
                cam_sd_token=cam_sd_t0,
                resize_hw=(H, W),
                classes_prefix=self.classes_prefix,
                max_K=self.max_K,
            )
        else:
            init_masks = torch.zeros(0, H, W, dtype=torch.float32)
            init_labels: List[int] = []

        traj = torch.tensor(e["target"]["future_xy"], dtype=torch.float32)  # [T_out,2]
        last = torch.tensor(e["target"]["last_xy"], dtype=torch.float32)    # [2]

        return {
            "frames": frames,             # [T_in,3,H,W]
            "lidar_maps": lidar_maps,     # [T_in,5,H,W]
            "traj": traj,                 # [T_out,2]
            "last_pos": last,             # [2]
            "init_masks": init_masks,     # [K,H,W]
            "init_labels": init_labels,   # list[int]
            "meta": e,
        }

# -------------------- collate for variable-K masks --------------------
def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = torch.stack([b["frames"] for b in batch], dim=0)      # [B,T,3,H,W]
    lidar  = torch.stack([b["lidar_maps"] for b in batch], dim=0)  # [B,T,5,H,W]
    traj   = torch.stack([b["traj"] for b in batch], dim=0)        # [B,T_out,2]
    last   = torch.stack([b["last_pos"] for b in batch], dim=0)    # [B,2]
    init_masks  = [b["init_masks"] for b in batch]
    init_labels = [b["init_labels"] for b in batch]
    meta = [b["meta"] for b in batch]
    return {
        "frames": frames,
        "lidar_maps": lidar,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }
