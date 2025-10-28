from typing import List, Dict, Any, Tuple  # Type hints for clarity/static checking
from PIL import Image 
from pyquaternion import Quaternion                  # Quaternion for rotations
import math                                          # Math utilities (atan, degrees, etc.)
import numpy as np


@staticmethod
def hfov_deg_from_fx(fx: float, crop_w: int) -> float:
    """
    Approximate horizontal FOV in degrees from focal length fx and crop width.
    HFOV = 2 * atan((W/2)/fx).
    """
    return 2.0 * math.degrees(math.atan((crop_w*0.5)/fx))

def yaw_from_sd(self, sd_token: str) -> float:
    """
    Estimate camera yaw angle in ego/world plane from calibrated sensor rotation.
    Returns yaw in radians (atan2 of projected +Z axis onto XY plane).
    """
    sd = self.nusc.get('sample_data', sd_token)                    # Sample data record
    cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])  # Calibrated sensor extrinsics
    R_ec = Quaternion(cs['rotation']).rotation_matrix             # 3x3 rotation (ego->camera or camera->ego per conv.)
    z = R_ec @ np.array([0.0, 0.0, 1.0])                          # Camera's +Z axis direction in parent frame
    return math.atan2(z[1], z[0])                                 # Yaw angle of that axis


def compute_physical_overlaps(self, row: Dict[str, Any], trip: List[str], crop_w: int) -> Tuple[int,int]:
    """
    Compute pixel overlaps between L-F and F-R crops by:
        - turning fx into HFOV per camera,
        - measuring yaw differences between cameras,
        - computing angular overlap,
        - converting overlap angle to pixels (min px/deg of the two cameras in each pair).
    """
    # Intrinsics matrices for triplet cameras
    KL = np.asarray(row["cams"][trip[0]]["intrinsics"], dtype=np.float64)
    KF = np.asarray(row["cams"][trip[1]]["intrinsics"], dtype=np.float64)
    KR = np.asarray(row["cams"][trip[2]]["intrinsics"], dtype=np.float64)

    # Focal lengths (assumes K[0,0] ~ fx in pixels)
    fxL = float(KL[0,0]); fxF = float(KF[0,0]); fxR = float(KR[0,0])

    # Horizontal FOV estimates (deg) for the resized crop width
    thL = hfov_deg_from_fx(fxL, crop_w)
    thF = hfov_deg_from_fx(fxF, crop_w)
    thR = hfov_deg_from_fx(fxR, crop_w)

    # Sample_data tokens (pick the first time step to estimate static overlaps)
    sdL = row["cams"][trip[0]]["sd_tokens"][0]
    sdF = row["cams"][trip[1]]["sd_tokens"][0]
    sdR = row["cams"][trip[2]]["sd_tokens"][0]

    # Camera yaw angles (radians)
    yawL = yaw_from_sd(self, sdL)
    yawF = yaw_from_sd(self, sdF)
    yawR = yaw_from_sd(self,sdR)

    # Helper: wrapped absolute angular difference
    def angdiff(a,b):
        d = a-b
        while d > math.pi: d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        return abs(d)

    # Pairwise yaw separations in degrees
    dLF = math.degrees(angdiff(yawL, yawF))
    dFR = math.degrees(angdiff(yawF, yawR))

    # Angular overlap between each pair: max(0, average half-FOVs - separation)
    ovLF_deg = max(0.0, 0.5*(thL+thF) - dLF)
    ovFR_deg = max(0.0, 0.5*(thF+thR) - dFR)

    # Pixels-per-degree for each camera crop (px/deg) at chosen crop width
    pdegL = crop_w / thL if thL>1e-6 else 0.0
    pdegF = crop_w / thF if thF>1e-6 else 0.0
    pdegR = crop_w / thR if thR>1e-6 else 0.0

    # Convert angular overlaps to pixels using the more conservative px/deg of the pair
    ov_lf_px = int(round(min(pdegL, pdegF) * ovLF_deg))
    ov_fr_px = int(round(min(pdegF, pdegR) * ovFR_deg))
    return ov_lf_px, ov_fr_px

def load_resize_arr(self, path: str, W: int, H: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load image from disk, remember original (W,H), resize to (W,H), return float32 array (optionally normalized).
    """
    img = Image.open(path).convert("RGB")          # Read and ensure 3-channel RGB
    ow, oh = img.size                              # Original width/height
    img = img.resize((W, H), resample=Image.BILINEAR)  # Resize to target per-cam crop (cw x H)
    arr = np.asarray(img, dtype=np.float32)        # (H, W, 3) float32
    if self.normalize:
        arr /= 255.0                               # Scale to [0,1] if requested
    return arr, (ow, oh)

def weights_1d( Wp: int, cw: int, ov_lf: int, ov_fr: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Build 1D horizontal blend weights for Left/Front/Right strips across the panorama width Wp.
    We use piecewise-linear ramps in the overlap regions; 1.0 in non-overlap interiors.
    """
    x = np.arange(Wp, dtype=np.float32)           # 0..Wp-1 coordinates
    wL = np.zeros(Wp, dtype=np.float32)           # Left weights
    wF = np.zeros(Wp, dtype=np.float32)           # Front weights
    wR = np.zeros(Wp, dtype=np.float32)           # Right weights

    # Segment boundaries:
    a0, a1 = 0, cw - ov_lf                         # Left-only interior
    b0, b1 = cw - ov_lf, cw                        # Left-Front overlap
    c0, c1 = cw, 2*cw - ov_lf - ov_fr              # Front-only interior
    d0, d1 = 2*cw - ov_lf - ov_fr, 2*cw - ov_lf    # Front-Right overlap
    e0 = 2*cw - ov_lf                               # Right-only interior start

    # Left-only: weight 1 for L
    wL[a0:a1] = 1.0

    # Left-Front overlap: linearly fade L->F
    if ov_lf > 0:
        t = (x[b0:b1] - b0) / max(ov_lf, 1)       # 0..1 across overlap
        wL[b0:b1] = 1.0 - t
        wF[b0:b1] = t

    # Front-only: weight 1 for F
    wF[c0:c1] = 1.0

    # Front-Right overlap: linearly fade F->R
    if ov_fr > 0:
        t2 = (x[d0:d1] - d0) / max(ov_fr, 1)      # 0..1 across overlap
        wF[d0:d1] = 1.0 - t2
        wR[d0:d1] = t2

    # Right-only: weight 1 for R
    wR[e0:Wp] = 1.0

    return wL, wF, wR


def compose_three(self, L: np.ndarray, F: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Blend three (H, cw, 3) images into a single (H, Wp, 3) panorama using the 1D weights.
    """
    Wp = self.W
    acc = np.zeros((self.H, Wp, 3), dtype=np.float32)   # Accumulator for weighted RGB
    wsum = np.zeros((self.H, Wp, 1), dtype=np.float32)  # Accumulator for sum of weights (per pixel)

    # Build horizontal 1D weights and expand to (H, Wp, 1) for broadcasting
    wL, wF, wR = weights_1d(Wp, self.cw, self.ov_lf_px, self.ov_fr_px)
    wL2 = np.repeat(wL[None, :, None], self.H, axis=0)
    wF2 = np.repeat(wF[None, :, None], self.H, axis=0)
    wR2 = np.repeat(wR[None, :, None], self.H, axis=0)

    # Paste Left crop at [0:cw]
    acc[:, 0:self.cw, :] += L * wL2[:, 0:self.cw, :]
    wsum[:, 0:self.cw, :] += wL2[:, 0:self.cw, :]

    # Paste Front crop shifted by left-overlap
    sF = self.cw - self.ov_lf_px
    acc[:, sF:sF + self.cw, :] += F * wF2[:, sF:sF + self.cw, :]
    wsum[:, sF:sF + self.cw, :] += wF2[:, sF:sF + self.cw, :]

    # Paste Right crop shifted by total overlaps
    sR = 2*self.cw - (self.ov_lf_px + self.ov_fr_px)
    acc[:, sR:sR + self.cw, :] += R * wR2[:, sR:sR + self.cw, :]
    wsum[:, sR:sR + self.cw, :] += wR2[:, sR:sR + self.cw, :]

    # Normalize by sum of weights to avoid darkening in overlaps
    comp = acc / np.clip(wsum, 1e-6, None)
    return comp
