# traj/datamodules.py
from typing import Any, Dict, List, Tuple, Optional

import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

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
    pts = view_points(corners_3d, cam_K, normalize=True)[:2, :].T  # (Nv, 2)
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
    classes_prefix: Tuple[str, ...] = ("vehicle.car","vehicle.truck","vehicle.bus","human.pedestrian"),
    max_K: int = 6,
):
    img_path, boxes, cam_K = nusc.get_sample_data(cam_sd_token, box_vis_level=0)
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

    masks: List[torch.Tensor] = []
    for b in selected:
        poly = _project_box_to_poly2d_safe(b, np.asarray(cam_K, dtype=np.float32), im_w, im_h)
        if poly is None:
            continue
        m = np.zeros((im_h, im_w), dtype=np.uint8)
        cv2.fillPoly(m, [poly], 1)
        if (im_h, im_w) != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(torch.from_numpy(m).float())

    if len(masks) == 0:
        M = torch.ones(1, H, W, dtype=torch.float32)  # fallback: full frame
        labels = [1]
    else:
        M = torch.stack(masks, dim=0)                  # [K,H,W]
        labels = list(range(1, M.size(0) + 1))
    return M, labels

# -------------------- Dataset --------------------
class NuScenesSeqLoader(Dataset):
    """
    Consumes agent-expanded index rows (one window × one target agent).
    Returns:
      frames      : [T_in, 3, H, W]
      traj        : [T_out, 2]  (global XY)
      last_pos    : [2]         (global XY)
      init_masks  : [K, H, W]   (variable K; CPU tensor)
      init_labels : list[int]   (1..K)
      meta        : dict
    """

    def __init__(
        self,
        index_path: str,
        nusc: NuScenes,
        resize_hw: Tuple[int, int] = (360, 640),
        classes_prefix: Tuple[str, ...] = ("vehicle.car","vehicle.truck","vehicle.bus","human.pedestrian"),
        max_K: int = 6,
    ):
        super().__init__()
        with open(index_path, "rb") as f:
            self.index: List[Dict[str, Any]] = pickle.load(f)

        assert len(self.index) > 0, "Empty index file."

        self.nusc = nusc
        self.resize_hw = resize_hw
        self.classes_prefix = classes_prefix
        self.max_K = max_K

        self.t_in = len(self.index[0]["obs_sample_tokens"])
        self.t_out = len(self.index[0]["fut_sample_tokens"])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.index[idx]
        frames = torch.stack([_load_and_resize_rgb(p, self.resize_hw) for p in e["img_paths"]], dim=0)  # [T_in,3,H,W]

        cam_sd_t0 = e["context"]["t0_cam_sd_token"]
        init_masks, init_labels = masks_from_t0(
            nusc=self.nusc,
            cam_sd_token=cam_sd_t0,
            resize_hw=self.resize_hw,
            classes_prefix=self.classes_prefix,
            max_K=self.max_K,
        )

        fut_xy = torch.tensor(e["target"]["future_xy"], dtype=torch.float32)   # [T_out,2]
        last_xy = torch.tensor(e["target"]["last_xy"], dtype=torch.float32)    # [2]

        return {
            "frames": frames,          # [T_in,3,H,W]
            "traj": fut_xy,            # [T_out,2] global
            "last_pos": last_xy,       # [2] global
            "init_masks": init_masks,  # [K,H,W] (CPU)
            "init_labels": init_labels,# list[int]
            "meta": e,
        }

# -------------------- collate: variable-K masks --------------------
def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = torch.stack([b["frames"] for b in batch], dim=0)   # [B,T,3,H,W]
    traj   = torch.stack([b["traj"] for b in batch], dim=0)     # [B,T_out,2]
    last   = torch.stack([b["last_pos"] for b in batch], dim=0) # [B,2]
    init_masks  = [b["init_masks"] for b in batch]              # list of [K_i,H,W]
    init_labels = [b["init_labels"] for b in batch]             # list[list[int]]
    meta = [b["meta"] for b in batch]
    return {
        "frames": frames,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }
