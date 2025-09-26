from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility


def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = torch.stack([b["frames"] for b in batch], dim=0)      # [B,T,3,H,W]
    traj   = torch.stack([b["traj"] for b in batch], dim=0)        # [B,T_out,2]
    last   = torch.stack([b["last_pos"] for b in batch], dim=0)    # [B,2]
    init_masks  = [b["init_masks"] for b in batch]
    init_labels = [b["init_labels"] for b in batch]
    meta = [b["meta"] for b in batch]
    return {
        "frames": frames,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }

class NuScenesSeqLoader(Dataset):
    def __init__(
        self,
        nusc: NuScenes,
        rows: List[Dict[str, Any]],
        out_size: Tuple[int, int] = (360, 640),   # (H, W) per camera view
        img_normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.nusc = nusc
        self.rows = rows
        self.H, self.W = int(out_size[0]), int(out_size[1])
        self.normalize = img_normalize
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _find_box_for_instance(nusc: NuScenes, sd_token: str, inst_tok: str):
        # was: box_vis_level=None  -> invalid
        _, boxes, _ = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
        for b in boxes:
            it = getattr(b, "instance_token", None)
            if it is None:
                ann = nusc.get('sample_annotation', b.token)
                it = ann['instance_token']
            if it == inst_tok:
                return b
        return None

    @staticmethod
    def _project_corners(K: np.ndarray, corners_cam: np.ndarray) -> np.ndarray:
        uvw = K @ corners_cam[:3, :]
        uv = uvw[:2, :] / np.clip(uvw[2:3, :], 1e-6, None)
        return uv  # [2,8]

    def _mask_from_box(
        self,
        nusc: NuScenes,
        sd_token: str,
        inst_tok: str,
        K: np.ndarray,
        target_hw: Tuple[int, int],
        orig_wh: Tuple[int, int],
    ) -> np.ndarray:
        box = self._find_box_for_instance(nusc, sd_token, inst_tok)
        if box is None:
            return np.zeros((target_hw[0], target_hw[1]), dtype=np.uint8)

        corners = box.corners()  # [3,8] in camera frame
        uv = self._project_corners(K, corners)  # [2,8]
        u_min, v_min = np.min(uv, axis=1)
        u_max, v_max = np.max(uv, axis=1)

        ow, oh = orig_wh
        u_min = float(np.clip(u_min, 0, ow - 1))
        v_min = float(np.clip(v_min, 0, oh - 1))
        u_max = float(np.clip(u_max, 0, ow - 1))
        v_max = float(np.clip(v_max, 0, oh - 1))
        if u_max <= u_min or v_max <= v_min:
            return np.zeros((target_hw[0], target_hw[1]), dtype=np.uint8)

        th, tw = target_hw
        sx = tw / ow
        sy = th / oh
        x0 = int(np.floor(u_min * sx))
        y0 = int(np.floor(v_min * sy))
        x1 = int(np.ceil (u_max * sx))
        y1 = int(np.ceil (v_max * sy))
        x0 = max(0, min(tw - 1, x0))
        y0 = max(0, min(th - 1, y0))
        x1 = max(0, min(tw, x1))
        y1 = max(0, min(th, y1))

        m = np.zeros((th, tw), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        return m

    def _load_and_resize(self, path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        img = Image.open(path).convert("RGB")
        ow, oh = img.size
        img = img.resize((self.W, self.H), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if self.normalize:
            arr /= 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).to(self.dtype)  # [3,H,W]
        return t, (ow, oh)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        obs_paths_grid: List[List[str]] = row["obs_cam_img_grid"]  # [T_in][C]
        cams: List[str] = row["cam_set"]
        T_in = len(obs_paths_grid)
        C = len(cams)

        frames_t: List[torch.Tensor] = []
        concat_masks_per_cam: List[np.ndarray] = []

        # Build init mask at t=0 for target across all cams
        inst_tok: str = row["target"]["agent_id"]
        init_masks_c = []
        for ci, cam in enumerate(cams):
            sd0 = row["cams"][cam]["sd_tokens"][0]
            K = np.asarray(row["cams"][cam]["intrinsics"], dtype=np.float32)
            # Load first frame for orig size to scale mask correctly
            img0 = Image.open(obs_paths_grid[0][ci]).convert("RGB")
            ow, oh = img0.size
            m = self._mask_from_box(self.nusc, sd0, inst_tok, K, (self.H, self.W), (ow, oh))
            init_masks_c.append(m)
        init_mask_full = np.concatenate(init_masks_c, axis=1)  # [H, W*C]
        init_mask_full_t = torch.from_numpy(init_mask_full[None, ...]).to(torch.uint8)  # [1,H,W*C]

        # Build frames over time: concat per-cam horizontally after resizing
        for t in range(T_in):
            per_cam_imgs = []
            for ci in range(C):
                img_t, _ = self._load_and_resize(obs_paths_grid[t][ci])
                per_cam_imgs.append(img_t)
            frame_t = torch.cat(per_cam_imgs, dim=2)  # [3,H,W*C]
            frames_t.append(frame_t)

        frames = torch.stack(frames_t, dim=0)  # [T,3,H,W*C]

        traj = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)          # [T_out,2]
        last_pos = torch.tensor(row["target"]["last_xy"], dtype=self.dtype)       # [2]

        sample = {
            "frames": frames,                       # [T_in,3,H,W*C]
            "traj": traj,                           # [T_out,2]
            "last_pos": last_pos,                   # [2]
            "init_masks": init_mask_full_t,       # K=1 -> list of [1,H,W*C]
            "init_labels": [1],                     # labels for XMem-style heads
            "meta": {
                "scene_name": row["scene_name"],
                "cams": cams,
                "start_sample_token": row["start_sample_token"],
                "anchor_cam": row["context"]["anchor_cam"],
                "W_concat": frames.shape[-1],
                "H": frames.shape[-2],
            },
        }
        return sample
