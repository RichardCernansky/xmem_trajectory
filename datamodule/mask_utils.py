from typing import  Tuple 
import numpy as np

from nuscenes.nuscenes import NuScenes               # NuScenes dataset API
from nuscenes.utils.geometry_utils import BoxVisibility  # Visibility enums for 3D boxes


@staticmethod
def find_box_for_instance(nusc: NuScenes, sd_token: str, inst_tok: str):
    """
    Given a sample_data token and an instance token, locate that 3D box in this camera's annotation set.
    Returns the NuScenes Box if found, else None.
    """
    _, boxes, _ = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)  # Fetch camera boxes
    for b in boxes:
        it = getattr(b, "instance_token", None)   # Some boxes store instance_token directly
        if it is None:
            ann = nusc.get('sample_annotation', b.token)  # Fallback: get via annotation object
            it = ann['instance_token']
        if it == inst_tok:                        # Match the target instance
            return b
    return None

@staticmethod
def project_corners(K: np.ndarray, corners_cam: np.ndarray) -> np.ndarray:
    """
    Project 3D corners in camera frame to pixel UV using intrinsics K.
    corners_cam: (3 or 4, N) with z>0; returns uv: (2, N)
    """
    uvw = K @ corners_cam[:3, :]                  # Apply pinhole model (no distortion)
    uv = uvw[:2, :] / np.clip(uvw[2:3, :], 1e-6, None)  # Divide by depth (avoid div-by-zero)
    return uv

def mask_from_box(
    self,
    nusc: NuScenes,
    sd_token: str,
    inst_tok: str,
    K: np.ndarray,
    target_hw: Tuple[int, int],                   # (H_t, W_t) of resized per-camera crop
    orig_wh: Tuple[int, int],                     # (W_orig, H_orig) of original image
) -> np.ndarray:
    """
    Create a binary mask by projecting the 3D box to 2D, taking its axis-aligned bbox in pixels,
    clamping to image bounds, and rescaling to the resized crop resolution.
    """
    box = find_box_for_instance(nusc, sd_token, inst_tok)  # Locate target box in this camera
    if box is None:
        return np.zeros((target_hw[0], target_hw[1]), dtype=np.uint8)  # No box -> empty mask
    corners = box.corners()                            # 3D box corners in camera frame (3 x 8)
    uv = project_corners(K, corners)             # 2 x 8 pixel coordinates
    u_min, v_min = np.min(uv, axis=1)                  # AABB min
    u_max, v_max = np.max(uv, axis=1)                  # AABB max

    ow, oh = orig_wh                                   # Original image size
    # Clamp AABB to original image bounds
    u_min = float(np.clip(u_min, 0, ow - 1))
    v_min = float(np.clip(v_min, 0, oh - 1))
    u_max = float(np.clip(u_max, 0, ow - 1))
    v_max = float(np.clip(v_max, 0, oh - 1))

    # Degenerate (no area) -> return empty
    if u_max <= u_min or v_max <= v_min:
        return np.zeros((target_hw[0], target_hw[1]), dtype=np.uint8)

    th, tw = target_hw                                 # Target resized crop H/W
    sx = tw / ow                                       # Horizontal scale factor
    sy = th / oh                                       # Vertical scale factor

    # Map original AABB to resized AABB
    x0 = int(np.floor(u_min * sx))
    y0 = int(np.floor(v_min * sy))
    x1 = int(np.ceil (u_max * sx))
    y1 = int(np.ceil (v_max * sy))

    # Clamp in resized coordinates
    x0 = max(0, min(tw - 1, x0))
    y0 = max(0, min(th - 1, y0))
    x1 = max(0, min(tw, x1))
    y1 = max(0, min(th, y1))

    # Fill rectangular region as mask=1
    m = np.zeros((th, tw), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def compose_masks(self, mL: np.ndarray, mF: np.ndarray, mR: np.ndarray) -> np.ndarray:
    """
    Compose three per-cam binary masks into the panorama coordinates using max() in overlaps.
    """
    out = np.zeros((self.H, self.W), dtype=np.uint8)

    # Left at [0:cw]
    out[:, 0:self.cw] = np.maximum(out[:, 0:self.cw], mL)

    # Front at [sF:sF+cw]
    sF = self.cw - self.ov_lf_px
    out[:, sF:sF + self.cw] = np.maximum(out[:, sF:sF + self.cw], mF)

    # Right at [sR:sR+cw]
    sR = 2*self.cw - (self.ov_lf_px + self.ov_fr_px)
    out[:, sR:sR + self.cw] = np.maximum(out[:, sR:sR + self.cw], mR)

    return out
