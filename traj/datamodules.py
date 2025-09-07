# traj/datamodules.py
import pickle
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


def _load_and_resize_rgb(path: str, hw: Tuple[int, int]) -> torch.Tensor:
    """Read BGR, convert to RGB, resize to (H,W), return float tensor [3,H,W] in [0,1]."""
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = hw
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    return t


def _project_box_to_poly2d(box, cam_intrinsic: np.ndarray, im_w: int, im_h: int) -> np.ndarray:
    """Project a 3D nuScenes Box to image polygon. Returns int32 polygon [N,2] clipped to image."""
    corners_3d = box.corners()  # (3,8)
    pts_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :].T  # (8,2)
    pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, im_w - 1)
    pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, im_h - 1)
    hull = cv2.convexHull(pts_2d.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.int32)
    return hull


def masks_from_t0(
    nusc: NuScenes,
    cam_sd_token: str,
    resize_hw: Tuple[int, int],
    classes_prefix: Tuple[str, ...] = ("vehicle.car", "vehicle.truck", "vehicle.bus"),
    max_K: int = 4,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build K binary masks [K,H,W] at t=0 by projecting 3D boxes to the image.
    If no eligible boxes are found, returns a single full-frame mask [1,H,W].
    Labels are [1..K] for XMem's label space.
    """
    img_path, boxes, K = nusc.get_sample_data(cam_sd_token, box_vis_level=0)
    im_bgr = cv2.imread(img_path)
    if im_bgr is None:
        raise FileNotFoundError(img_path)
    im_h, im_w = im_bgr.shape[:2]
    H, W = resize_hw

    # pick relevant classes; sort by distance (closest first)
    selected = []
    for b in boxes:
        if any(b.name.startswith(p) for p in classes_prefix):
            dist = float(np.linalg.norm(b.center))
            selected.append((dist, b))
    selected.sort(key=lambda x: x[0])
    selected = [b for _, b in selected[:max_K]]

    masks = []
    for b in selected:
        poly = _project_box_to_poly2d(b, np.array(K, dtype=np.float32), im_w, im_h)
        m = np.zeros((im_h, im_w), dtype=np.uint8)
        if len(poly) >= 3:
            cv2.fillPoly(m, [poly], 1)
        if (im_h, im_w) != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(torch.from_numpy(m).float())

    if len(masks) == 0:
        M = torch.ones(1, H, W, dtype=torch.float32)  # fallback: full-frame single mask
        labels = [1]
    else:
        M = torch.stack(masks, dim=0)                 # [K,H,W]
        labels = list(range(1, M.size(0) + 1))

    return M, labels


class NuScenesSeqLoader(Dataset):
    """
    Dataset backed by a prebuilt index (train_index.pkl / val_index.pkl).
    Loads observed frames and builds masks for t=0 on the fly.

    Each index entry contains:
      - scene_name, start_sample_token
      - obs_sample_tokens (len = T_in), fut_sample_tokens (len = T_out)
      - cam_sd_tokens (len = T_in), img_paths (len = T_in), intrinsics (len = T_in)
    """

    def __init__(
        self,
        index_path: str,
        nusc: Optional[NuScenes] = None,          # required for masks (use the same version/dataroot as when indexing)
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

        if self.nusc is None:
            raise ValueError("NuScenes object is required to build masks on the fly (pass nusc=NuScenes(...)).")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.index[idx]
        H, W = self.resize_hw

        # --- Load observed frames (T_in) ---
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

        # --- Ground-truth trajectory placeholders (replace with your extraction) ---
        T_out = len(entry["fut_sample_tokens"])
        traj = torch.zeros(T_out, 2, dtype=torch.float32)   # [T_out,2]
        last_pos = torch.zeros(2, dtype=torch.float32)      # [2]

        return {
            "frames": frames,           # [T_in,3,H,W]
            "traj": traj,               # [T_out,2]
            "last_pos": last_pos,       # [2]
            "init_masks": init_masks,   # [K,H,W]  (CPU tensor)
            "init_labels": init_labels, # e.g. [1] or [1,2,...,K]
            "meta": entry               # keep the tokens for future GT building
        }


# -------- Optional: a collate_fn that can handle variable-K masks --------
def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Default PyTorch collate can't stack masks if K varies per sample.
    This collate:
      - stacks frames to [B,T,3,H,W]
      - keeps init_masks as a list of [K_i,H,W] tensors
      - keeps init_labels as a list of lists
      - stacks traj and last_pos
    """
    B = len(batch)
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # [B,T,3,H,W]
    traj   = torch.stack([b["traj"] for b in batch], dim=0)    # [B,T_out,2]
    last   = torch.stack([b["last_pos"] for b in batch], dim=0)# [B,2]
    init_masks  = [b["init_masks"] for b in batch]             # list of [K_i,H,W]
    init_labels = [b["init_labels"] for b in batch]            # list of list[int]
    meta = [b["meta"] for b in batch]
    return {
        "frames": frames,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }
