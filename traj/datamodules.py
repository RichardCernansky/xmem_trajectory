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
    """
    Read BGR from disk, convert to RGB, resize to (H,W), return float tensor [3,H,W] in [0,1].
    """
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
    """
    Project a nuScenes 3D Box (already in the camera frame from get_sample_data)
    into a valid 2D polygon. Returns int32 array [N,2] or None if invalid.
    """
    # 3x8 corners in camera frame
    corners_3d = box.corners()  # (3, 8)

    # keep corners with positive depth
    z = corners_3d[2, :]
    valid = z > 1e-6
    if valid.sum() < 3:
        return None

    corners_3d = corners_3d[:, valid]  # (3, Nv) Nv>=3

    # project and normalize
    pts = view_points(corners_3d, cam_K, normalize=True)[:2, :].T  # (Nv, 2)

    # remove non-finite points
    if not np.isfinite(pts).all():
        pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None

    # clip to image and build convex hull
    pts[:, 0] = np.clip(pts[:, 0], 0, im_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, im_h - 1)

    # OpenCV wants (N,1,2) and float32/int32
    pts_cv = np.ascontiguousarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_cv).reshape(-1, 2).astype(np.int32)
    if hull.shape[0] < 3:
        return None
    return hull



def masks_from_t0(
    nusc, cam_sd_token: str, resize_hw, classes_prefix=("vehicle.car","vehicle.truck","vehicle.bus"),
    max_K: int = 4,
):
    img_path, boxes, cam_K = nusc.get_sample_data(cam_sd_token, box_vis_level=0)  # cam_K is 3x3
    im_bgr = cv2.imread(img_path)
    if im_bgr is None:
        raise FileNotFoundError(img_path)
    im_h, im_w = im_bgr.shape[:2]
    H, W = resize_hw

    # pick classes and sort by distance (closest first)
    selected = []
    for b in boxes:
        if any(b.name.startswith(p) for p in classes_prefix):
            dist = float(np.linalg.norm(b.center))
            selected.append((dist, b))
    selected.sort(key=lambda x: x[0])
    selected = [b for _, b in selected[:max_K]]

    masks = []
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
        M = torch.ones(1, H, W, dtype=torch.float32)  # fallback: full-frame
        labels = [1]
    else:
        M = torch.stack(masks, dim=0)
        labels = list(range(1, M.size(0) + 1))
    return M, labels


# -------------------- Dataset consuming your NEW index format --------------------

class NuScenesSeqLoader(Dataset):
    """
    Dataset backed by a prebuilt index (train_index.pkl / val_index.pkl) of the form:

    {
      "scene_name": "scene-0061",
      "start_sample_token": "...",
      "obs_sample_tokens": ["tok1", "tok2", ...],   # len = t_in
      "fut_sample_tokens": ["tokX", ...],           # len = t_out
      "cam_sd_tokens": ["sd_tok1", "sd_tok2", ...], # len = t_in (observed frames only)
      "img_paths": ["/abs/path/CAM_FRONT/....jpg", ...], # len = t_in
      "intrinsics": [[...3x3...], ...],             # len = t_in (each a 3x3 list)
      "cam": "CAM_FRONT"
    }

    Returns per item:
      frames:      [T_in, 3, H, W]   (float in [0,1])
      traj:        [T_out, 2]        (placeholder zeros; replace with your GT extraction)
      last_pos:    [2]               (placeholder zeros)
      init_masks:  [K, H, W]         (CPU tensor, K can vary per sample)
      init_labels: list[int]         (1..K)
      meta:        dict              (the full index entry)
    """

    def __init__(
        self,
        index_path: str,
        nusc: NuScenes,
        resize_hw: Tuple[int, int] = (360, 640),
        classes_prefix: Tuple[str, ...] = ("vehicle.car", "vehicle.truck", "vehicle.bus"),
        max_K: int = 4,
    ):
        super().__init__()
        with open(index_path, "rb") as f:
            self.index: List[Dict[str, Any]] = pickle.load(f)

        self.nusc = nusc
        self.resize_hw = resize_hw
        self.classes_prefix = classes_prefix
        self.max_K = max_K

        # Basic checks to catch format mismatches early
        assert len(self.index) > 0, "Empty index file."
        req_keys = {
            "scene_name", "start_sample_token", "obs_sample_tokens", "fut_sample_tokens",
            "cam_sd_tokens", "img_paths", "intrinsics", "cam"
        }
        missing = req_keys - set(self.index[0].keys())
        if missing:
            raise KeyError(f"Index entries missing keys: {missing}")

        # Cache lengths for convenience
        self.t_in = len(self.index[0]["obs_sample_tokens"])
        self.t_out = len(self.index[0]["fut_sample_tokens"])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.index[idx]
        H, W = self.resize_hw

        # --- Load observed frames (T_in) from pre-resolved image paths ---
        frames = [_load_and_resize_rgb(p, self.resize_hw) for p in entry["img_paths"]]
        frames = torch.stack(frames, dim=0)  # [T_in,3,H,W]

        # --- Build masks at t=0 (from first observed camera sample_data token) ---
        cam_sd_t0 = entry["cam_sd_tokens"][0]
        init_masks, init_labels = masks_from_t0(
            nusc=self.nusc,
            cam_sd_token=cam_sd_t0,
            resize_hw=self.resize_hw,
            classes_prefix=self.classes_prefix,
            max_K=self.max_K,
        )  # [K,H,W], list[int]

        # --- Ground-truth trajectory placeholders (replace with your own logic) ---
        traj = torch.zeros(self.t_out, 2, dtype=torch.float32)   # [T_out,2]
        last_pos = torch.zeros(2, dtype=torch.float32)           # [2]

        return {
            "frames": frames,             # [T_in,3,H,W]
            "traj": traj,                 # [T_out,2]
            "last_pos": last_pos,         # [2]
            "init_masks": init_masks,     # [K,H,W]  (CPU tensor)
            "init_labels": init_labels,   # e.g. [1] or [1,2,...,K]
            "meta": entry                 # keep the tokens (useful for GT extraction/debug)
        }


# -------------------- collate for variable-K masks --------------------

def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Default PyTorch collate can't stack masks if K varies per sample.
    This collate:
      - stacks frames to [B,T,3,H,W]
      - stacks traj to [B,T_out,2] and last_pos to [B,2]
      - keeps init_masks as a *list* of [K_i,H,W] tensors
      - keeps init_labels as a *list* of lists
      - carries meta as a list of dicts
    """
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



