from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN_SCENES, val as VAL_SCENES
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion

from trainer.utils import open_config
from index_nuscenes.visibility_check import _check_sequence_mask_visibility


DEFAULT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]  # default 3-camera subset
DEFAULT_LIDAR = "LIDAR_TOP"                                        # only LiDAR stream in nuScenes

def _print_filtering_stats(stats: Dict):
    """Print filtering statistics."""
    print("\n" + "="*70)
    print("INDEX FILTERING STATISTICS (MASK-BASED)")
    print("="*70)
    print(f"Total candidate sequences:                 {stats['total_candidates']}")
    print(f"  ✓ Accepted:                              {stats['accepted']} ({100*stats['accepted']/max(1,stats['total_candidates']):.1f}%)")
    print(f"\nRejection reasons:")
    print(f"  ✗ First frame < 500 pixels:              {stats['rejected_first_frame_too_small']}")
    print(f"  ✗ Too many invisible frames:             {stats['rejected_too_many_invisible']}")
    print(f"  ✗ Consecutive invisible frames:          {stats['rejected_consecutive_invisible']}")
    print(f"  ✗ Other:                                 {stats['rejected_other']}")
    print("="*70 + "\n")

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



def build_agent_sequence_index(config_path: str) -> List[Dict]:
    """Build index using configuration from JSON file with mask-based filtering."""
    config = open_config(config_path)
    
    # Initialize nuScenes
    nusc = NuScenes(
        version=config["dataset"]["version"],
        dataroot=config["dataset"]["dataroot"],
        verbose=True
    )
    
    # Extract config values
    seq_cfg = config["sequence"]
    sensor_cfg = config["sensors"]
    filter_cfg = config["filtering"]
    output_cfg = config["output"]
    
    cameras = sensor_cfg["cameras"]
    if not cameras:
        cameras = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    
    anchor_cam = cameras[0]
    
    # Parse visibility level
    visibility_map = {
        "NONE": BoxVisibility.NONE,
        "ANY": BoxVisibility.ANY,
        "ALL": BoxVisibility.ALL
    }
    visibility = visibility_map.get(filter_cfg.get("camera_visibility", "ANY"), BoxVisibility.ANY)
    
    # Scene filtering
    splits = config["dataset"].get("splits")
    if splits == "train":
        wanted = set(TRAIN_SCENES)
        scenes = [s for s in nusc.scene if s["name"] in wanted]
    elif splits == "val":
        wanted = set(VAL_SCENES)
        scenes = [s for s in nusc.scene if s["name"] in wanted]
    else:
        scenes = nusc.scene
    
    rows: List[Dict] = []
    t_in = seq_cfg["t_in"]
    t_out = seq_cfg["t_out"]
    total = t_in + t_out
    stride = seq_cfg["stride"]
    
    # Stats tracking
    stats = {
        "total_candidates": 0,
        "rejected_first_frame_too_small": 0,
        "rejected_too_many_invisible": 0,
        "rejected_consecutive_invisible": 0,
        "rejected_other": 0,
        "accepted": 0
    }
    
    for sc in scenes:
        scene_name = sc["name"]
        tokens = _scene_tokens(nusc, scene_name)
        if len(tokens) < total:
            continue
        
        for start in range(0, len(tokens) - total + 1, stride):
            obs_tokens = tokens[start : start + t_in]
            fut_tokens = tokens[start + t_in : start + t_in + t_out]
            
            # Collect camera data
            cams_block = {cam: {"sd_tokens": [], "img_paths": []} for cam in cameras}
            intrinsics_once = {}
            obs_cam_img_grid = []
            
            for st in obs_tokens:
                cam_info_map, ordered_paths = _cams_sd_and_imgs(nusc, st, cameras)
                obs_cam_img_grid.append(ordered_paths)
                for cam in cameras:
                    cams_block[cam]["sd_tokens"].append(cam_info_map[cam]["sd_token"])
                    cams_block[cam]["img_paths"].append(cam_info_map[cam]["img_path"])
                    if cam not in intrinsics_once:
                        intrinsics_once[cam] = cam_info_map[cam]["K"].tolist()
            
            # LiDAR sweep aggregation
            lidar_keyframe_sd_tokens = []
            lidar_sd_tokens_per_obs = []
            lidar_pc_paths_per_obs = []
            lidar_dt_per_obs = []
            
            for st in obs_tokens:
                toks, paths, dts = _lidar_prev_sweeps_sd_and_paths(
                    nusc, st, sensor_cfg["lidar_sensor"],
                    n_sweeps=sensor_cfg["num_lidar_sweeps"],
                    max_age_s=sensor_cfg["max_lidar_sweep_age_s"]
                )
                lidar_keyframe_sd_tokens.append(toks[0])
                lidar_sd_tokens_per_obs.append(toks)
                lidar_pc_paths_per_obs.append(paths)
                lidar_dt_per_obs.append(dts)
            
            # Ego anchor frame
            sd_anchor = cams_block[anchor_cam]["sd_tokens"][-1]
            t_w, R_w = _ego_pose_from_sd(nusc, sd_anchor)
            R_we_2x2 = R_w[:2, :2].T
            t_w_2 = t_w[:2]
            
            def world_xy_to_ego_xy(xy_world):
                p = np.asarray(xy_world, dtype=np.float32) - t_w_2
                q = R_we_2x2 @ p
                return [float(q[0]), float(q[1])]
            
            ann_last = _ann_by_instance(nusc, obs_tokens[-1])
            if not ann_last:
                continue
            
            for inst_tok, last_ann in ann_last.items():
                stats["total_candidates"] += 1
                
                # Class filter
                name = last_ann.get("category_name", "")
                if not any(name.startswith(p) for p in filter_cfg["class_prefixes"]):
                    continue
                
                # Future trajectory
                fut_xy_e, valid = [], 0
                for ft in fut_tokens:
                    m = _ann_by_instance(nusc, ft)
                    ann_f = m.get(inst_tok, None)
                    if ann_f is None:
                        break
                    fut_xy_e.append(world_xy_to_ego_xy(_xy_from_ann_global(ann_f)))
                    valid += 1
                
                min_future = seq_cfg.get("min_future") or t_out
                if valid < min_future:
                    continue
                
                # Speed filter
                min_speed = seq_cfg.get("min_speed_mps", 0.0)
                if min_speed > 0 and valid >= 2:
                    j = min(3, len(fut_xy_e) - 1)
                    x3, y3 = fut_xy_e[j]
                    dist = (x3**2 + y3**2) ** 0.5
                    approx_speed = dist / 1.5
                    if approx_speed < min_speed:
                        continue
                
                last_xy_e = world_xy_to_ego_xy(_xy_from_ann_global(last_ann))
                
                # Mask-based filtering
                mask_req = filter_cfg.get("mask_requirements", {})
                if mask_req.get("enabled", True):
                    is_valid, vis_stats = _check_sequence_mask_visibility(
                        nusc, obs_tokens, lidar_keyframe_sd_tokens, inst_tok, config
                    )
                    
                    if not is_valid:
                        reason = vis_stats["rejection_reason"]
                        if "first_frame_too_small" in reason:
                            stats["rejected_first_frame_too_small"] += 1
                        elif "too_many_invisible" in reason:
                            stats["rejected_too_many_invisible"] += 1
                        elif "consecutive_invisible" in reason:
                            stats["rejected_consecutive_invisible"] += 1
                        else:
                            stats["rejected_other"] += 1
                        continue
                
                # Build row
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
                        "sensor": sensor_cfg["lidar_sensor"],
                        "keyframe_sd_tokens": lidar_keyframe_sd_tokens,
                        "sd_tokens": lidar_sd_tokens_per_obs,
                        "pc_paths": lidar_pc_paths_per_obs,
                        "dt_s": lidar_dt_per_obs,
                        "nsweeps": sensor_cfg["num_lidar_sweeps"],
                        "max_sweep_age_s": sensor_cfg["max_lidar_sweep_age_s"]
                    },
                    "target": {
                        "agent_id": inst_tok,
                        "last_xy": last_xy_e,
                        "future_xy": fut_xy_e,
                        "frame": "ego_xy"
                    },
                    "context": {
                        "anchor_cam": anchor_cam,
                        "anchor_sd_token": sd_anchor,
                        "anchor_lidar_sd_token": lidar_keyframe_sd_tokens[-1]
                    },
                    "dataroot": config["dataset"]["dataroot"]
                }
                
                rows.append(row)
                stats["accepted"] += 1
                
                throttle = output_cfg.get("throttle_max_rows")
                if throttle and len(rows) >= throttle:
                    if output_cfg["verbose_stats"]:
                        _print_filtering_stats(stats)
                    return rows
    
    if output_cfg["verbose_stats"]:
        _print_filtering_stats(stats)
    
    return rows