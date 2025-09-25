from typing import List, Dict, Optional, Tuple
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN_SCENES, val as VAL_SCENES
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import BoxVisibility

# ---------- helpers ----------

def _is_instance_in_camera(nusc: NuScenes, cam_sd_token: str, inst_tok: str,
                           vis: BoxVisibility = BoxVisibility.ANY) -> bool:
    _, boxes, _ = nusc.get_sample_data(cam_sd_token, box_vis_level=vis)
    for b in boxes:
        # Prefer direct match if present
        it = getattr(b, "instance_token", None)
        if it is None:
            # Map annotation token -> instance_token
            ann = nusc.get('sample_annotation', b.token)
            it = ann['instance_token']
        if it == inst_tok:
            return True
    return False

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
    sd = nusc.get("sample_data", sd_tok)
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.asarray(calib["camera_intrinsic"], dtype=np.float32)
    img_path = nusc.get_sample_data_path(sd_tok)
    return sd_tok, img_path, K

def _lidar_sd_and_bin(nusc: NuScenes, sample_token: str):
    sd_tok = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
    return sd_tok, nusc.get_sample_data_path(sd_tok)

def _ann_by_instance(nusc: NuScenes, sample_token: str) -> Dict[str, dict]:
    """instance_token -> sample_annotation at this sample (if present)."""
    s = nusc.get("sample", sample_token)
    out = {}
    for ann_tok in s["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        out[ann["instance_token"]] = ann
    return out

def _xy_from_ann_global(ann: dict) -> Tuple[float, float]:
    x, y, _ = ann["translation"]
    return float(x), float(y)

def _ego_pose_from_sd(nusc: NuScenes, sd_token: str):
    """Return (t_world: (3,), R_world(3x3)) for the ego pose attached to a sample_data."""
    sd = nusc.get("sample_data", sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    t = np.asarray(ego["translation"], dtype=np.float32)            # (3,)
    R = Quaternion(ego["rotation"]).rotation_matrix.astype(np.float32)  # (3,3)
    return t, R

def _T_from_calib(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """4x4 transform from calibrated_sensor (sensor-in-ego)."""
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    t = np.asarray(cs["translation"], dtype=np.float32)
    R = Quaternion(cs["rotation"]).transformation_matrix
    R[:3, 3] = t
    return R.astype(np.float32)  # T_ego_sensor

def _pose_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """4x4 ego pose (ego-in-world) for a sample_data."""
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    t = np.asarray(ep["translation"], dtype=np.float32)
    R = Quaternion(ep["rotation"]).transformation_matrix
    R[:3, 3] = t
    return R.astype(np.float32)  # T_world_ego

def _compute_T_cam_lidar(nusc: NuScenes, cam_sd_token: str, lidar_sd_token: str) -> np.ndarray:
    """
    Return 4x4 transform from LiDAR to camera: T_cam_lidar.
    T_world_cam = T_world_ego_cam * T_ego_cam
    T_world_lid = T_world_ego_lid * T_ego_lid
    T_cam_lidar = (T_world_cam)^-1 * T_world_lid
    """
    T_world_ego_cam = _pose_4x4(nusc, cam_sd_token)
    T_world_ego_lid = _pose_4x4(nusc, lidar_sd_token)
    T_ego_cam = _T_from_calib(nusc, cam_sd_token)
    T_ego_lid = _T_from_calib(nusc, lidar_sd_token)
    T_world_cam = T_world_ego_cam @ T_ego_cam
    T_world_lid = T_world_ego_lid @ T_ego_lid
    T_cam_world = np.linalg.inv(T_world_cam)
    return (T_cam_world @ T_world_lid).astype(np.float32)

# ---------- main builder (EGO-CENTRIC) ----------

def build_agent_sequence_index(
    nusc: NuScenes,
    splits: Optional[str] = None,              # "train" | "val" | None
    scene_names: Optional[List[str]] = None,   # if provided, overrides splits
    cam: str = "CAM_FRONT",
    t_in: int = 8,
    t_out: int = 10,
    stride: int = 1,
    min_future: Optional[int] = None,          # require at least N future steps (<= t_out)
    min_speed_mps: float = 0.0,                # skip near-stationary targets if > 0
    class_prefixes: Tuple[str, ...] = ("vehicle."),
    dataroot: Optional[str] = None,
) -> List[Dict]:
    """
    Each row = (window, target agent) with coordinates in the EGO frame anchored at obs[t_in-1].
    Also includes per-observation LiDAR paths and T_cam_lidar matrices for projection/fusion.
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

            # camera & lidar data for observed frames
            cam_sd_tokens, img_paths, intrinsics = [], [], []
            lidar_sd_tokens, lidar_paths, T_cam_lidar = [], [], []
            for st in obs_tokens:
                cam_sd, img, K = _cam_sd_and_img(nusc, st, cam)
                lid_sd, lid_path = _lidar_sd_and_bin(nusc, st)
                cam_sd_tokens.append(cam_sd)
                img_paths.append(img.replace("\\", "/"))
                intrinsics.append(K.tolist())
                lidar_sd_tokens.append(lid_sd)
                lidar_paths.append(lid_path.replace("\\", "/"))
                T_cam_lidar.append(_compute_T_cam_lidar(nusc, cam_sd, lid_sd).tolist())

            # ego frame anchor = ego pose of LAST observed camera sample_data
            sd_anchor = cam_sd_tokens[-1]
            t_w, R_w = _ego_pose_from_sd(nusc, sd_anchor)
            R_we_2x2 = R_w[:2, :2].T  # inverse rotation (2D)
            t_w_2 = t_w[:2]

            def world_xy_to_ego_xy(xy_world: Tuple[float, float]) -> List[float]:
                p = np.asarray(xy_world, dtype=np.float32) - t_w_2
                q = R_we_2x2 @ p
                return [float(q[0]), float(q[1])]

            # candidates from last observed sample
            ann_last = _ann_by_instance(nusc, obs_tokens[-1])
            if not ann_last:
                continue

            for inst_tok, last_ann in ann_last.items():
                name = last_ann.get("category_name", "")
                if not any(name.startswith(p) for p in class_prefixes):
                    continue

                # future XY in ego coords
                fut_xy_e, valid = [], 0
                for ft in fut_tokens:
                    m = _ann_by_instance(nusc, ft)
                    ann_f = m.get(inst_tok, None)
                    if ann_f is None:
                        break
                    fut_xy_e.append(world_xy_to_ego_xy(_xy_from_ann_global(ann_f)))
                    valid += 1
                if valid < min_future:
                    continue

                # stationary filter (approx over ~1.5s @2Hz)
                if min_speed_mps > 0 and valid >= 2:
                    # Use ego distance of the 3rd future point as rough proxy
                    j = min(3, len(fut_xy_e)-1)
                    x3, y3 = fut_xy_e[j]
                    dist = (x3**2 + y3**2) ** 0.5
                    approx_speed = dist / 1.5
                    if approx_speed < min_speed_mps:
                        continue

                last_xy_e = world_xy_to_ego_xy(_xy_from_ann_global(last_ann))

                # Ensure target is in the chosen camera at t=0 (first observed frame)
                if not _is_instance_in_camera(nusc, cam_sd_tokens[0], inst_tok):
                    continue

                rows.append({
                    "scene_name": scene_name,
                    "start_sample_token": obs_tokens[0],
                    "obs_sample_tokens": obs_tokens,
                    "fut_sample_tokens": fut_tokens,
                    "cam": cam,

                    "cam_sd_tokens": cam_sd_tokens,     # len = t_in
                    "img_paths": img_paths,             # len = t_in
                    "intrinsics": intrinsics,           # len = t_in, 3x3 each
                    "lidar_sd_tokens": lidar_sd_tokens, # len = t_in
                    "lidar_paths": lidar_paths,         # len = t_in
                    "T_cam_lidar": T_cam_lidar,         # len = t_in, 4x4 each

                    "target": {
                        "agent_id": inst_tok,
                        "last_xy": last_xy_e,            # EGO frame at t_in-1
                        "future_xy": fut_xy_e,           # EGO frame, len = t_out
                        "frame": "ego_xy"
                    },
                    "context": {
                        "anchor_sd_token": sd_anchor     # ego frame anchor (optional)
                    },
                    "dataroot": dataroot or ""           # helps loaders rebuild NuScenes if needed
                })

                if len(rows) == 2000:
                    return rows


    return rows
