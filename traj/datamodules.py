from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion
import math


def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    traj   = torch.stack([b["traj"] for b in batch], dim=0)
    last   = torch.stack([b["last_pos"] for b in batch], dim=0)
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
        out_size: Tuple[int, int]=(160, 704),
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
        self.trip = self._choose_triplet(self.rows[0]["cam_set"], self.pano_triplet)
        self.ov_lf_px, self.ov_fr_px = self._compute_physical_overlaps(self.rows[0], self.trip, self.cw)
        self.ov_lf_px = int(np.clip(self.ov_lf_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.ov_fr_px = int(np.clip(self.ov_fr_px, int(self.min_or*self.cw), int(self.max_or*self.cw)))
        self.W = 3*self.cw - (self.ov_lf_px + self.ov_fr_px)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _choose_triplet(all_cams: List[str], prefer: List[str]) -> List[str]:
        have = [c for c in prefer if c in all_cams]
        if len(have) == 3: return have
        for c in all_cams:
            if c not in have and len(have) < 3:
                have.append(c)
        return have[:3]

    @staticmethod
    def _find_box_for_instance(nusc: NuScenes, sd_token: str, inst_tok: str):
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
        return uv

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
        corners = box.corners()
        uv = self._project_corners(K, corners)
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

    def _load_resize_arr(self, path: str, W: int, H: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        img = Image.open(path).convert("RGB")
        ow, oh = img.size
        img = img.resize((W, H), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if self.normalize:
            arr /= 255.0
        return arr, (ow, oh)

    @staticmethod
    def _hfov_deg_from_fx(fx: float, crop_w: int) -> float:
        return 2.0 * math.degrees(math.atan((crop_w*0.5)/fx))

    def _yaw_from_sd(self, sd_token: str) -> float:
        sd = self.nusc.get('sample_data', sd_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        R_ec = Quaternion(cs['rotation']).rotation_matrix
        z = R_ec @ np.array([0.0, 0.0, 1.0])
        return math.atan2(z[1], z[0])

    def _compute_physical_overlaps(self, row: Dict[str, Any], trip: List[str], crop_w: int) -> Tuple[int,int]:
        KL = np.asarray(row["cams"][trip[0]]["intrinsics"], dtype=np.float64)
        KF = np.asarray(row["cams"][trip[1]]["intrinsics"], dtype=np.float64)
        KR = np.asarray(row["cams"][trip[2]]["intrinsics"], dtype=np.float64)
        fxL = float(KL[0,0]); fxF = float(KF[0,0]); fxR = float(KR[0,0])
        thL = self._hfov_deg_from_fx(fxL, crop_w)
        thF = self._hfov_deg_from_fx(fxF, crop_w)
        thR = self._hfov_deg_from_fx(fxR, crop_w)
        sdL = row["cams"][trip[0]]["sd_tokens"][0]
        sdF = row["cams"][trip[1]]["sd_tokens"][0]
        sdR = row["cams"][trip[2]]["sd_tokens"][0]
        yawL = self._yaw_from_sd(sdL)
        yawF = self._yaw_from_sd(sdF)
        yawR = self._yaw_from_sd(sdR)
        def angdiff(a,b):
            d = a-b
            while d > math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            return abs(d)
        dLF = math.degrees(angdiff(yawL, yawF))
        dFR = math.degrees(angdiff(yawF, yawR))
        ovLF_deg = max(0.0, 0.5*(thL+thF) - dLF)
        ovFR_deg = max(0.0, 0.5*(thF+thR) - dFR)
        pdegL = crop_w / thL if thL>1e-6 else 0.0
        pdegF = crop_w / thF if thF>1e-6 else 0.0
        pdegR = crop_w / thR if thR>1e-6 else 0.0
        ov_lf_px = int(round(min(pdegL, pdegF) * ovLF_deg))
        ov_fr_px = int(round(min(pdegF, pdegR) * ovFR_deg))
        return ov_lf_px, ov_fr_px

    def _weights_1d(self, Wp: int, cw: int, ov_lf: int, ov_fr: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        x = np.arange(Wp, dtype=np.float32)
        wL = np.zeros(Wp, dtype=np.float32)
        wF = np.zeros(Wp, dtype=np.float32)
        wR = np.zeros(Wp, dtype=np.float32)
        a0, a1 = 0, cw - ov_lf
        b0, b1 = cw - ov_lf, cw
        c0, c1 = cw, 2*cw - ov_lf - ov_fr
        d0, d1 = 2*cw - ov_lf - ov_fr, 2*cw - ov_lf
        e0 = 2*cw - ov_lf
        wL[a0:a1] = 1.0
        if ov_lf > 0:
            t = (x[b0:b1] - b0) / max(ov_lf, 1)
            wL[b0:b1] = 1.0 - t
            wF[b0:b1] = t
        wF[c0:c1] = 1.0
        if ov_fr > 0:
            t2 = (x[d0:d1] - d0) / max(ov_fr, 1)
            wF[d0:d1] = 1.0 - t2
            wR[d0:d1] = t2
        wR[e0:Wp] = 1.0
        return wL, wF, wR

    def _compose_three(self, L: np.ndarray, F: np.ndarray, R: np.ndarray) -> np.ndarray:
        Wp = self.W
        acc = np.zeros((self.H, Wp, 3), dtype=np.float32)
        wsum = np.zeros((self.H, Wp, 1), dtype=np.float32)
        wL, wF, wR = self._weights_1d(Wp, self.cw, self.ov_lf_px, self.ov_fr_px)
        wL2 = np.repeat(wL[None, :, None], self.H, axis=0)
        wF2 = np.repeat(wF[None, :, None], self.H, axis=0)
        wR2 = np.repeat(wR[None, :, None], self.H, axis=0)
        acc[:, 0:self.cw, :] += L * wL2[:, 0:self.cw, :]
        wsum[:, 0:self.cw, :] += wL2[:, 0:self.cw, :]
        sF = self.cw - self.ov_lf_px
        acc[:, sF:sF + self.cw, :] += F * wF2[:, sF:sF + self.cw, :]
        wsum[:, sF:sF + self.cw, :] += wF2[:, sF:sF + self.cw, :]
        sR = 2*self.cw - (self.ov_lf_px + self.ov_fr_px)
        acc[:, sR:sR + self.cw, :] += R * wR2[:, sR:sR + self.cw, :]
        wsum[:, sR:sR + self.cw, :] += wR2[:, sR:sR + self.cw, :]
        comp = acc / np.clip(wsum, 1e-6, None)
        return comp

    def _compose_masks(self, mL: np.ndarray, mF: np.ndarray, mR: np.ndarray) -> np.ndarray:
        out = np.zeros((self.H, self.W), dtype=np.uint8)
        out[:, 0:self.cw] = np.maximum(out[:, 0:self.cw], mL)
        sF = self.cw - self.ov_lf_px
        out[:, sF:sF + self.cw] = np.maximum(out[:, sF:sF + self.cw], mF)
        sR = 2*self.cw - (self.ov_lf_px + self.ov_fr_px)
        out[:, sR:sR + self.cw] = np.maximum(out[:, sR:sR + self.cw], mR)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        obs_paths_grid: List[List[str]] = row["obs_cam_img_grid"]
        cams_all: List[str] = row["cam_set"]
        trip = self._choose_triplet(cams_all, self.trip)
        idxs = [cams_all.index(c) for c in trip]
        T_in = len(obs_paths_grid)
        frames_t: List[torch.Tensor] = []

        inst_tok: str = row["target"]["agent_id"]
        m0L = m0F = m0R = None
        owL = ohL = owF = ohF = owR = ohR = None
        KL = KF = KR = None
        sdL = sdF = sdR = None

        for t in range(T_in):
            pL = obs_paths_grid[t][idxs[0]]
            pF = obs_paths_grid[t][idxs[1]]
            pR = obs_paths_grid[t][idxs[2]]
            imL, (owL, ohL) = self._load_resize_arr(pL, self.cw, self.H)
            imF, (owF, ohF) = self._load_resize_arr(pF, self.cw, self.H)
            imR, (owR, ohR) = self._load_resize_arr(pR, self.cw, self.H)
            comp = self._compose_three(imL, imF, imR)
            frames_t.append(torch.from_numpy(comp).permute(2, 0, 1).to(self.dtype))

            if t == 0:
                sdL = row["cams"][trip[0]]["sd_tokens"][0]
                sdF = row["cams"][trip[1]]["sd_tokens"][0]
                sdR = row["cams"][trip[2]]["sd_tokens"][0]
                KL = np.asarray(row["cams"][trip[0]]["intrinsics"], dtype=np.float32)
                KF = np.asarray(row["cams"][trip[1]]["intrinsics"], dtype=np.float32)
                KR = np.asarray(row["cams"][trip[2]]["intrinsics"], dtype=np.float32)
                m0L = self._mask_from_box(self.nusc, sdL, inst_tok, KL, (self.H, self.cw), (owL, ohL))
                m0F = self._mask_from_box(self.nusc, sdF, inst_tok, KF, (self.H, self.cw), (owF, ohF))
                m0R = self._mask_from_box(self.nusc, sdR, inst_tok, KR, (self.H, self.cw), (owR, ohR))

        frames = torch.stack(frames_t, dim=0)

        if m0L is None:
            init_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        else:
            init_mask = self._compose_masks(m0L, m0F, m0R)
        init_mask_t = torch.from_numpy(init_mask[None, ...]).to(torch.uint8)

        traj = torch.tensor(row["target"]["future_xy"], dtype=self.dtype)
        last_pos = torch.tensor(row["target"]["last_xy"], dtype=self.dtype)

        sample = {
            "frames": frames,
            "traj": traj,
            "last_pos": last_pos,
            "init_masks": init_mask_t,
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
        return sample
