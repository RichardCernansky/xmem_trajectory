from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN_SCENES, val as VAL_SCENES
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion

DEFAULT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]  # default 3-camera subset
DEFAULT_LIDAR = "LIDAR_TOP"                                        # only LiDAR stream in nuScenes

def _is_instance_in_camera(nusc: NuScenes, cam_sd_token: str, inst_tok: str,
                           vis: BoxVisibility = BoxVisibility.ANY) -> bool:
    _, boxes, _ = nusc.get_sample_data(cam_sd_token, box_vis_level=vis)
    for b in boxes:
        it = getattr(b, "instance_token", None)
        if it is None:
            ann = nusc.get('sample_annotation', b.token)  # fallback to annotation if missing on box
            it = ann['instance_token']
        if it == inst_tok:
            return True
    return False

def _is_instance_in_any_cam(nusc: NuScenes, sample_token: str, inst_tok: str,
                            cameras: List[str],
                            vis: BoxVisibility = BoxVisibility.ANY) -> bool:
    s = nusc.get("sample", sample_token)
    for cam in cameras:
        sd_tok = s["data"][cam]
        if _is_instance_in_camera(nusc, sd_tok, inst_tok, vis=vis):
            return True
    return False

def _scene_tokens(nusc: NuScenes, scene_name: str) -> List[str]:
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    toks = []
    tok = scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        toks.append(tok)
        tok = s["next"] if s["next"] else None
    return toks

def _rel_to_root(nusc: NuScenes, p: str) -> str:
    # store paths relative to dataset root for portability across machines
    return str(Path(p).resolve().relative_to(Path(nusc.dataroot).resolve()))

def _cam_sd_and_img(nusc: NuScenes, sample_token: str, cam: str):
    s = nusc.get("sample", sample_token)
    sd_tok = s["data"][cam]
    sd = nusc.get("sample_data", sd_tok)
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.asarray(calib["camera_intrinsic"], dtype=np.float32)  # pinhole intrinsics
    abs_img = nusc.get_sample_data_path(sd_tok).replace("\\", "/")
    rel_img = _rel_to_root(nusc, abs_img)                        # keep relative path
    return sd_tok, rel_img, K

def _cams_sd_and_imgs(nusc: NuScenes, sample_token: str, cameras: List[str]):
    out = {}
    ordered_paths = []
    for cam in cameras:
        sd_tok, rel_img, K = _cam_sd_and_img(nusc, sample_token, cam)
        out[cam] = {"sd_token": sd_tok, "img_path": rel_img, "K": K}
        ordered_paths.append(rel_img)                            # grid row for this timestep
    return out, ordered_paths

def _lidar_sd_and_path(nusc: NuScenes, sample_token: str, lidar_sensor: str):
    s = nusc.get("sample", sample_token)
    sd_tok = s["data"][lidar_sensor]                             # keyframe LiDAR at sample time
    abs_path = nusc.get_sample_data_path(sd_tok).replace("\\", "/")
    rel_path = _rel_to_root(nusc, abs_path)
    return sd_tok, rel_path

def _lidar_prev_sweeps_sd_and_paths(nusc: NuScenes, sample_token: str, lidar_sensor: str,
                                    n_sweeps: int = 12, max_age_s: Optional[float] = 0.6):
    # collect up to n_sweeps LiDAR sample_data tokens: keyframe + previous non-keyframe sweeps
    s = nusc.get("sample", sample_token)
    sd0 = nusc.get("sample_data", s["data"][lidar_sensor])       # anchor (keyframe) at t0
    ts0 = sd0["timestamp"]
    toks = [sd0["token"]]                                        # include keyframe first
    paths = [_rel_to_root(nusc, nusc.get_sample_data_path(sd0["token"]).replace("\\", "/"))]
    deltas = [0.0]                                               # time offset (s) relative to t0
    cur = sd0
    while cur["prev"] and len(toks) < n_sweeps:
        cur = nusc.get("sample_data", cur["prev"])               # follow prev chain (higher-rate sweeps)
        if max_age_s is not None and (ts0 - cur["timestamp"]) / 1e6 > max_age_s:
            break                                                # clamp temporal window to avoid smear
        toks.append(cur["token"])
        paths.append(_rel_to_root(nusc, nusc.get_sample_data_path(cur["token"]).replace("\\", "/")))
        deltas.append(-(ts0 - cur["timestamp"]) / 1e6)           # negative: in the past
    return toks, paths, deltas

def _ann_by_instance(nusc: NuScenes, sample_token: str) -> Dict[str, dict]:
    s = nusc.get("sample", sample_token)
    out = {}
    for ann_tok in s["anns"]:
        ann = nusc.get("sample_annotation", ann_tok)
        out[ann["instance_token"]] = ann                          # map instance -> its latest ann
    return out

def _xy_from_ann_global(ann: dict) -> Tuple[float, float]:
    x, y, _ = ann["translation"]
    return float(x), float(y)

def _ego_pose_from_sd(nusc: NuScenes, sd_token: str):
    sd = nusc.get("sample_data", sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    t = np.asarray(ego["translation"], dtype=np.float32)          # world->ego translation
    R = Quaternion(ego["rotation"]).rotation_matrix.astype(np.float32)  # world->ego rotation
    return t, R

def _pose_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # homogeneous transform: global <- ego at this sample_data timestamp
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T = Quaternion(ep["rotation"]).transformation_matrix
    T[:3, 3] = np.asarray(ep["translation"], dtype=np.float32)
    return T.astype(np.float32)

def _T_sensor_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    # sensor frame -> ego frame at that timestamp
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    T = Quaternion(cs["rotation"]).transformation_matrix
    T[:3, 3] = np.asarray(cs["translation"], dtype=np.float32)
    return T.astype(np.float32)

def _T_ego_to_sensor_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    return np.linalg.inv(_T_sensor_to_ego_4x4(nusc, sd_token)).astype(np.float32)

def _T_ego_to_global_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    return _pose_4x4(nusc, sd_token)

def _T_global_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    return np.linalg.inv(_pose_4x4(nusc, sd_token)).astype(np.float32)

def _T_cam_from_lidar_4x4(nusc: NuScenes, cam_sd_token: str, lidar_sd_token: str) -> np.ndarray:
    # compose: LiDAR -> ego(lidar_time) -> global -> ego(cam_time) -> cam
    T_cam_from_ego = _T_ego_to_sensor_4x4(nusc, cam_sd_token)
    T_ego_cam_from_global = _T_global_to_ego_4x4(nusc, cam_sd_token)
    T_global_from_ego_lidar = _T_ego_to_global_4x4(nusc, lidar_sd_token)
    T_ego_from_lidar = _T_sensor_to_ego_4x4(nusc, lidar_sd_token)
    T = T_cam_from_ego @ T_ego_cam_from_global @ T_global_from_ego_lidar @ T_ego_from_lidar
    return T.astype(np.float32)

def build_agent_sequence_index(
    nusc: NuScenes,
    cameras: Optional[List[str]] = None,
    splits: Optional[str] = None,
    scene_names: Optional[List[str]] = None,
    t_in: int = 8,
    t_out: int = 10,
    stride: int = 1,
    min_future: Optional[int] = None,
    min_speed_mps: float = 0.0,
    class_prefixes: Tuple[str, ...] = ("vehicle.",),
    dataroot: Optional[str] = None,
    visibility: BoxVisibility = BoxVisibility.ANY,
    throttle_max_rows: Optional[int] = None,
    lidar_sensor: str = DEFAULT_LIDAR,
    compute_lidar_transforms: bool = False,
    num_lidar_sweeps: int = 12,                   # aggregate 12 past sweeps (incl. keyframe)
    max_lidar_sweep_age_s: Optional[float] = 0.6  # cap time window to ~0.6 s
) -> List[Dict]:
    if cameras is None or len(cameras) == 0:
        cameras = list(DEFAULT_CAMS)
    anchor_cam = cameras[0]                        # define anchor camera for ego frame alignment
    if min_future is None:
        min_future = t_out

    # scene filtering by split/name
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
            continue  # not enough frames for this (t_in + t_out) window

        for start in range(0, len(tokens) - total + 1, stride):
            obs_tokens = tokens[start : start + t_in]
            fut_tokens = tokens[start + t_in : start + t_in + t_out]

            # collect camera sample_data tokens and image paths for each observed timestep
            cams_block: Dict[str, Dict[str, list]] = {cam: {"sd_tokens": [], "img_paths": []} for cam in cameras}
            intrinsics_once: Dict[str, List[List[float]]] = {}    # cache intrinsics per camera
            obs_cam_img_grid: List[List[str]] = []                # grid of image paths [t][cam]

            for i, st in enumerate(obs_tokens):
                cam_info_map, ordered_paths = _cams_sd_and_imgs(nusc, st, cameras)
                obs_cam_img_grid.append(ordered_paths)
                for cam in cameras:
                    cams_block[cam]["sd_tokens"].append(cam_info_map[cam]["sd_token"])
                    cams_block[cam]["img_paths"].append(cam_info_map[cam]["img_path"])
                    if cam not in intrinsics_once:
                        intrinsics_once[cam] = cam_info_map[cam]["K"].tolist()

            # LiDAR sweep aggregation (past-only, causal)
            lidar_keyframe_sd_tokens: List[str] = []              # one keyframe token per observed timestep
            lidar_sd_tokens_per_obs: List[List[str]] = []         # per-timestep list of sweep tokens
            lidar_pc_paths_per_obs: List[List[str]] = []          # per-timestep list of sweep paths
            lidar_dt_per_obs: List[List[float]] = []              # per-timestep list of Δt (s) per sweep

            for st in obs_tokens:
                toks, paths, dts = _lidar_prev_sweeps_sd_and_paths(
                    nusc, st, lidar_sensor, n_sweeps=num_lidar_sweeps, max_age_s=max_lidar_sweep_age_s
                )
                lidar_keyframe_sd_tokens.append(toks[0])          # first is always the keyframe at st
                lidar_sd_tokens_per_obs.append(toks)
                lidar_pc_paths_per_obs.append(paths)
                lidar_dt_per_obs.append(dts)

            # precompute transforms: cam <- lidar_sweep for each cam/obs/sweep
            T_cam_from_lidar: Dict[str, List[List[List[List[float]]]]] = {cam: [] for cam in cameras}
            if compute_lidar_transforms:
                for i in range(len(obs_tokens)):                  # iterate over observed timesteps
                    for cam in cameras:
                        cam_sd = cams_block[cam]["sd_tokens"][i]
                        mats = []
                        for lidar_sd in lidar_sd_tokens_per_obs[i]:
                            T = _T_cam_from_lidar_4x4(nusc, cam_sd, lidar_sd)
                            mats.append(T.tolist())
                        T_cam_from_lidar[cam].append(mats)

            # define ego anchor using the last observed frame of the anchor camera
            sd_anchor = cams_block[anchor_cam]["sd_tokens"][-1]
            t_w, R_w = _ego_pose_from_sd(nusc, sd_anchor)
            R_we_2x2 = R_w[:2, :2].T                              # world->ego (2D plane)
            t_w_2 = t_w[:2]

            def world_xy_to_ego_xy(xy_world: Tuple[float, float]) -> List[float]:
                p = np.asarray(xy_world, dtype=np.float32) - t_w_2
                q = R_we_2x2 @ p
                return [float(q[0]), float(q[1])]

            ann_last = _ann_by_instance(nusc, obs_tokens[-1])     # agents present at last observed time
            if not ann_last:
                continue

            for inst_tok, last_ann in ann_last.items():
                name = last_ann.get("category_name", "")
                if not any(name.startswith(p) for p in class_prefixes):
                    continue                                      # class filter (e.g., vehicles only)

                # collect future trajectory in anchor ego frame
                fut_xy_e, valid = [], 0
                for ft in fut_tokens:
                    m = _ann_by_instance(nusc, ft)
                    ann_f = m.get(inst_tok, None)
                    if ann_f is None:
                        break
                    fut_xy_e.append(world_xy_to_ego_xy(_xy_from_ann_global(ann_f)))
                    valid += 1
                if valid < (min_future or t_out):
                    continue                                      # require minimum future length

                if min_speed_mps > 0 and valid >= 2:
                    j = min(3, len(fut_xy_e) - 1)                 # rough speed from early horizon
                    x3, y3 = fut_xy_e[j]
                    dist = (x3**2 + y3**2) ** 0.5
                    approx_speed = dist / 1.5                     # ~0.15 s per sweep * 10 sweeps ≈ 1.5 s heuristic
                    if approx_speed < min_speed_mps:
                        continue

                last_xy_e = world_xy_to_ego_xy(_xy_from_ann_global(last_ann))

                # ensure target is visible in at least one camera at the start of observation
                if not _is_instance_in_any_cam(nusc, obs_tokens[0], inst_tok, cameras, vis=visibility):
                    continue

                # assemble index row
                row = {
                    "scene_name": scene_name,
                    "start_sample_token": obs_tokens[0],
                    "obs_sample_tokens": obs_tokens,
                    "fut_sample_tokens": fut_tokens,
                    "cam_set": list(cameras),
                    "cams": {
                        cam: {
                            "sd_tokens": cams_block[cam]["sd_tokens"],
                            "img_paths": cams_block[cam]["img_paths"],
                            "intrinsics": intrinsics_once[cam]
                        } for cam in cameras
                    },
                    "obs_cam_img_grid": obs_cam_img_grid,
                    "lidar": {
                        "sensor": lidar_sensor,
                        "keyframe_sd_tokens": lidar_keyframe_sd_tokens,  # one per observed timestep
                        "sd_tokens": lidar_sd_tokens_per_obs,            # sweeps per timestep (incl. keyframe)
                        "pc_paths": lidar_pc_paths_per_obs,              # relative .pcd/.bin paths per sweep
                        "dt_s": lidar_dt_per_obs,                        # Δt of each sweep relative to t0
                        "nsweeps": num_lidar_sweeps,
                        "max_sweep_age_s": max_lidar_sweep_age_s,
                        "T_cam_from_lidar": T_cam_from_lidar            # [cam][t][sweep] 4x4 matrices
                    },
                    "target": {
                        "agent_id": inst_tok,
                        "last_xy": last_xy_e,
                        "future_xy": fut_xy_e,
                        "frame": "ego_xy"                                # all XY in anchor ego frame
                    },
                    "context": {
                        "anchor_cam": anchor_cam,
                        "anchor_sd_token": sd_anchor,
                        "anchor_lidar_sd_token": lidar_keyframe_sd_tokens[-1] if lidar_keyframe_sd_tokens else None
                    },
                    "dataroot": dataroot or ""
                }

                rows.append(row)
                if throttle_max_rows and len(rows) >= throttle_max_rows:
                    return rows

    return rows
