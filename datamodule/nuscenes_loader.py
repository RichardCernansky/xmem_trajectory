# ---- Imports and typing ----
from typing import List, Dict, Any, Tuple, Optional  # Type hints for clarity/static checking
import numpy as np                                   # Numerical arrays and math
import torch                                         # Tensors for PyTorch training pipelines
from torch.utils.data import Dataset                 # Base class to build custom datasets

from nuscenes.nuscenes import NuScenes               # NuScenes dataset API
from .image_utils import compute_physical_overlaps, load_resize_arr, compose_three
from .mask_utils import mask_from_box, compose_masks


class NuScenesLoader(Dataset):
    """
    Dataset that:
      - loads a short image sequence per sample (multi-time steps),
      - composes a panoramic strip from three cameras (Left, Front, Right),
      - builds an initial 2D mask (axis-aligned box) for a tracked instance,
      - outputs frames + trajectory labels for training.
    """

    def __init__(
        self,
        nusc: NuScenes,                               # NuScenes API handle (preloaded externally)
        rows: List[Dict[str, Any]],                   # Prebuilt metadata rows (one per sample)
        out_size: Tuple[int, int]=(384, 640),         # Target (H, W) for composed panorama (H used; W computed)
        img_normalize: bool=True,                     # If True -> scale RGB to [0,1]
        dtype: torch.dtype=torch.float32,             # Tensor dtype for frames/labels
        crop_w: int=320,                              # Per-camera crop width after resize
        pano_triplet: Optional[List[str]]=None,       # Preferred triplet of camera names
        min_overlap_ratio: float=0.15,                # Clamp lower bound of overlaps (as fraction of crop_w)
        max_overlap_ratio: float=0.6,                 # Clamp upper bound of overlaps (as fraction of crop_w)
    ):
        self.nusc = nusc
        self.rows = rows
        self.H = int(out_size[0])                     # Final panorama height (pixels)
        self.normalize = img_normalize
        self.dtype = dtype
        self.cw = int(crop_w)                         # Width of each camera crop in the panorama
        # Default to a front pano triplet if none provided
        self.pano_triplet = pano_triplet or ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT"]
        self.min_or = float(min_overlap_ratio)
        self.max_or = float(max_overlap_ratio)

        # Fix the triplet order for this loader based on the first row's available cameras
        self.trip = self.pano_triplet

        # Estimate physical overlaps (in px) between (Left,Front) and (Front,Right) given intrinsics+yaw
        self.ov_lf_px, self.ov_fr_px = compute_physical_overlaps(self, self.rows[0], self.trip, self.cw)

        # Clamp overlaps to reasonable fractional bounds of crop width to avoid degenerate blends
        self.ov_lf_px = int(np.clip(self.ov_lf_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.ov_fr_px = int(np.clip(self.ov_fr_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))

        # Final panorama width after subtracting the two overlaps from 3 * crop_w
        self.W = 3*self.cw - (self.ov_lf_px + self.ov_fr_px)

    def __len__(self) -> int:
        # Number of samples equals number of metadata rows provided
        return len(self.rows)
    

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Build one training sample:
          - Compose T frames into panoramas (T, C, H, W),
          - Build initial instance mask in panorama coords,
          - Provide trajectory labels and metadata.
        """
        row = self.rows[idx]                                     # Sample metadata
        obs_paths_grid: List[List[str]] = row["obs_cam_img_grid"]# Paths grid: time x cams -> filepath
        cams_all: List[str] = row["cam_set"]                     # All camera names available for this sample

        # Choose the triplet (consistent with loader's preferred ordering & availability)
        trip =cams_all
        idxs = [cams_all.index(c) for c in trip]                 # Indices into per-time camera lists

        T_in = len(obs_paths_grid)                               # Number of observed time steps
        frames_t: List[torch.Tensor] = []                        # Will accumulate T tensors (C,H,W)

        # Target instance id we will mask
        inst_tok: str = row["target"]["agent_id"]

        # Placeholders we fill at t==0 (used for initial mask projection)
        m0L = m0F = m0R = None
        owL = ohL = owF = ohF = owR = ohR = None
        KL = KF = KR = None
        sdL = sdF = sdR = None

        # Loop over time steps, load three images, compose panorama per time t
        for t in range(T_in):
            pL = obs_paths_grid[t][idxs[0]]                      # Left image path at time t
            pF = obs_paths_grid[t][idxs[1]]                      # Front image path at time t
            pR = obs_paths_grid[t][idxs[2]]                      # Right image path at time t

            # Load+resize each to (H, cw)
            imL, (owL, ohL) = load_resize_arr(self, pL, self.cw, self.H)  # (H, cw, 3)
            imF, (owF, ohF) = load_resize_arr(self, pF, self.cw, self.H)
            imR, (owR, ohR) = load_resize_arr(self, pR, self.cw, self.H)

            # Alpha-blend into one (H, W, 3) panorama
            comp = compose_three(self, imL, imF, imR)

            # Convert to torch (C,H,W) and append
            frames_t.append(torch.from_numpy(comp).permute(2, 0, 1).to(self.dtype))

            # On the first frame, prepare initial 2D masks for the tracked agent from 3 cameras
            if t == 0:
                sdL = row["cams"][trip[0]]["sd_tokens"][0]       # sample_data token for Left cam
                sdF = row["cams"][trip[1]]["sd_tokens"][0]       #                 for Front cam
                sdR = row["cams"][trip[2]]["sd_tokens"][0]       #                 for Right cam

                KL = np.asarray(row["cams"][trip[0]]["intrinsics"], dtype=np.float32)  # K_left
                KF = np.asarray(row["cams"][trip[1]]["intrinsics"], dtype=np.float32)  # K_front
                KR = np.asarray(row["cams"][trip[2]]["intrinsics"], dtype=np.float32)  # K_right

                # Build per-cam binary masks at (H, cw) by projecting 3D box -> AABB -> resized crop
                m0L = mask_from_box(self, self.nusc, sdL, inst_tok, KL, (self.H, self.cw), (owL, ohL))
                m0F = mask_from_box(self, self.nusc, sdF, inst_tok, KF, (self.H, self.cw), (owF, ohF))
                m0R = mask_from_box(self, self.nusc, sdR, inst_tok, KR, (self.H, self.cw), (owR, ohR))

        # Stack time dimension -> (T, C, H, W)
        frames = torch.stack(frames_t, dim=0)

        # Compose initial panorama mask (1,H,W) uint8; if no mask was built, return zeros
        if m0L is None:
            init_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        else:
            init_mask = compose_masks(self, m0L, m0F, m0R)
        init_mask_t = torch.from_numpy(init_mask[None, ...]).to(torch.uint8)  # Add channel dim

        # Trajectory supervision: future positions and the last observed position
        traj = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)     # (T_future, 2)
        last_pos = torch.tensor(row["target"]["last_xy"], dtype=self.dtype)   # (2,)

        # Package sample
        sample = {
            "frames": frames,                          # (T_in, 3, H, W)
            "traj": traj,                              # (T_future, 2)
            "last_pos": last_pos,                      # (2,)
            "init_masks": init_mask_t,                 # (1, H, W) uint8
            "init_labels": [1],                        # Dummy label list (e.g., 'object present')
            "meta": {
                "scene_name": row["scene_name"],       # Scene identifier
                "cams": trip,                          # Cameras used for panorama
                "start_sample_token": row["start_sample_token"],  # Anchor token
                "pano_w": self.W,                      # Panorama width (computed)
                "H": self.H,                           # Panorama height
                "overlap_lf": self.ov_lf_px,           # Overlap L-F in pixels
                "overlap_fr": self.ov_fr_px,           # Overlap F-R in pixels
                "crop_w": self.cw,                     # Per-camera crop width
            },
        }
        return sample
