# nuscenes_index_agents.py
from typing import List, Dict, Optional, Tuple
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN_SCENES, val as VAL_SCENES
from pyquaternion import Quaternion

# ---------- helpers ----------

def _scene_tokens(nusc: NuScenes, scene_name: str) -> List[str]:
    """Ordered sample tokens for a scene name."""
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    toks = []
    tok = scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        toks.append(tok)
        tok = s["next"] if s["next"] else None
    return toks

def _cam_sd_and_img(nusc: NuScenes, sample_token: str, cam: str):
    """(sample_data token, img_path, intrinsics 3x3) for camera at a sample."""
    s = nusc.get("sample", sample_token)
    sd_tok = s["data"][cam]
    img_path, _, K = nusc.get_sample_data(sd_tok, box_vis_level=0)
    return sd_tok, img_path, np.asarray(K, dtype=np.float32)

def _ann_by_instance(nusc: NuScenes, sample_token: str) -> Dict[str, dict]:
    """instance_token -> sample_annotation at this sample (if present)."""
    s = nusc.get("sample", sample_token)
    out = {}
    for ann_tok in s["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        out[ann["instance_token"]] = ann
    return out

def _xy_from_ann_global(ann: dict):
    x, y, _ = ann["translation"]
    return float(x), float(y)

def _ego_pose_from_sd(nusc: NuScenes, sd_token: str):
    """Return (t_world: (3,), R_world(3x3)) for the ego pose attached to a sample_data."""
    sd = nusc.get("sample_data", sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    t = np.asarray(ego["translation"], dtype=np.float32)  # (3,)
    R = Quaternion(ego["rotation"]).rotation_matrix.astype(np.float32)  # (3,3)
    return t, R

# ---------- main builder (EGO-CENTRIC) ----------

def build_agent_sequence_index(
    nusc: NuScenes,
    splits: Optional[str] = None,              # "train" | "val" | None
    scene_names: Optional[List[str]] = None,   # if provided, overrides splits
    cam: str = "CAM_FRONT",
    t_in: int = 8,
    t_out: int = 30,
    stride: int = 1,
    min_future: Optional[int] = None,          # require at least N future steps (<= t_out)
    min_speed_mps: float = 0.0,                # skip near-stationary targets if > 0
    class_prefixes: Tuple[str, ...] = ("vehicle.", "human.pedestrian"),
) -> List[Dict]:
    """
    Build a flat list where EACH ROW is (window, target agent).
    Coordinates are saved in the EGO frame at the last observed timestep:
      - last_xy, future_xy are expressed in ego(x,y) (meters).
      - The ego frame is defined by the CAM rig pose at obs[t_in-1].

    Note: you can still switch to agent-centric later by subtracting target last_xy.
    """
    if min_future is None:
        min_future = t_out

    # choose scenes
    if scene_names is not None:
        wanted = set(scene_names)
        scenes = [s for s in nusc.scene if s["name"] in wanted]
    elif splits == "train":
        wanted = set(TRAIN_SCENES)
        scenes = [s for s in nusc.scene if s["name"] in wanted]
    elif splits == "val":
        wanted = set(VAL_SCENES)
        scenes = [s for s in nusc.scene if s["name"] in wanted]
    else:
        scenes = nusc.scene

    rows: List[Dict] = []
    total = t_in + t_out

    for sc in scenes:
        scene_name = sc["name"]
        tokens = _scene_tokens(nusc, scene_name)
        if len(tokens) < total:
            continue

        # slide window
        for start in range(0, len(tokens) - total + 1, stride):
            obs_tokens = tokens[start : start + t_in]
            fut_tokens = tokens[start + t_in : start + t_in + t_out]

            # camera data for observed frames
            cam_sd_tokens, img_paths, intrinsics = [], [], []
            for st in obs_tokens:
                sd_tok, path, K = _cam_sd_and_img(nusc, st, cam)
                cam_sd_tokens.append(sd_tok)
                img_paths.append(path)
                intrinsics.append(K.tolist())

            # EGO frame anchor: use the ego pose of the LAST observed camera sample_data
            sd_anchor = cam_sd_tokens[-1]
            t_w, R_w = _ego_pose_from_sd(nusc, sd_anchor)  # ego pose in world at t_in-1
            # world->ego rotation (2D) and translation
            R_we_2x2 = R_w[:2, :2].T               # inverse of R_w (2D block)
            t_w_2 = t_w[:2]                        # world translation

            def world_xy_to_ego_xy(xy_world: Tuple[float, float]) -> List[float]:
                p = np.asarray(xy_world, dtype=np.float32) - t_w_2     # translate to ego origin
                q = R_we_2x2 @ p                                       # rotate into ego axes
                return [float(q[0]), float(q[1])]

            # candidate targets from last observed sample
            ann_last = _ann_by_instance(nusc, obs_tokens[-1])
            if not ann_last:
                continue

            for inst_tok, last_ann in ann_last.items():
                # coarse class filter
                name = last_ann.get("category_name", "")
                if not any(name.startswith(p) for p in class_prefixes):
                    continue

                # collect future XY (world), then convert to ego
                fut_xy_e = []
                valid = 0
                for ft in fut_tokens:
                    m = _ann_by_instance(nusc, ft)
                    ann_f = m.get(inst_tok, None)
                    if ann_f is None:
                        break
                    fut_xy_e.append(world_xy_to_ego_xy(_xy_from_ann_global(ann_f)))
                    valid += 1

                if valid < min_future:
                    continue

                # stationary filter (~1.5s at 2Hz), in WORLD before transform (either is fine)
                if min_speed_mps > 0 and valid >= 2:
                    x0, y0, _ = last_ann["translation"]
                    x3, y3 = fut_xy_e[min(3, len(fut_xy_e)-1)]  # NOTE: this is ego now
                    # Use ego distances; same threshold meaning (meters / 1.5s)
                    dist = (x3**2 + y3**2) ** 0.5
                    approx_speed = dist / 1.5
                    if approx_speed < min_speed_mps:
                        continue

                last_xy_e = world_xy_to_ego_xy(_xy_from_ann_global(last_ann))

                rows.append({
                    "scene_name": scene_name,
                    "start_sample_token": obs_tokens[0],
                    "obs_sample_tokens": obs_tokens,
                    "fut_sample_tokens": fut_tokens,
                    "cam": cam,
                    "cam_sd_tokens": cam_sd_tokens,   # len = t_in
                    "img_paths": img_paths,           # len = t_in
                    "intrinsics": intrinsics,         # len = t_in, 3x3 each

                    "target": {
                        "agent_id": inst_tok,
                        "last_xy": last_xy_e,         # EGO frame at t_in-1
                        "future_xy": fut_xy_e,        # EGO frame, len = t_out
                        "frame": "ego_xy"
                    },

                    "context": {
                        "t0_cam_sd_token": cam_sd_tokens[0],  # for on-the-fly masks
                        "anchor_sd_token": sd_anchor          # ego frame anchor (optional)
                    }
                })

    return rows
