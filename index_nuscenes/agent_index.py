from typing import List, Dict, Optional, Tuple
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train as TRAIN_SCENES, val as VAL_SCENES
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion

DEFAULT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
DEFAULT_LIDAR = "LIDAR_TOP"

def _is_instance_in_camera(nusc: NuScenes, cam_sd_token: str, inst_tok: str,
                           vis: BoxVisibility = BoxVisibility.ANY) -> bool:
    _, boxes, _ = nusc.get_sample_data(cam_sd_token, box_vis_level=vis)
    for b in boxes:
        it = getattr(b, "instance_token", None)
        if it is None:
            ann = nusc.get('sample_annotation', b.token)
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

def _cam_sd_and_img(nusc: NuScenes, sample_token: str, cam: str):
    s = nusc.get("sample", sample_token)
    sd_tok = s["data"][cam]
    sd = nusc.get("sample_data", sd_tok)
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.asarray(calib["camera_intrinsic"], dtype=np.float32)
    img_path = nusc.get_sample_data_path(sd_tok)
    return sd_tok, img_path.replace("\\", "/"), K

def _cams_sd_and_imgs(nusc: NuScenes, sample_token: str, cameras: List[str]):
    out = {}
    ordered_paths = []
    for cam in cameras:
        sd_tok, img, K = _cam_sd_and_img(nusc, sample_token, cam)
        out[cam] = {"sd_token": sd_tok, "img_path": img, "K": K}
        ordered_paths.append(img)
    return out, ordered_paths

def _lidar_sd_and_path(nusc: NuScenes, sample_token: str, lidar_sensor: str):
    s = nusc.get("sample", sample_token)
    sd_tok = s["data"][lidar_sensor]
    path = nusc.get_sample_data_path(sd_tok)
    return sd_tok, path.replace("\\", "/")

def _ann_by_instance(nusc: NuScenes, sample_token: str) -> Dict[str, dict]:
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
    sd = nusc.get("sample_data", sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    t = np.asarray(ego["translation"], dtype=np.float32)
    R = Quaternion(ego["rotation"]).rotation_matrix.astype(np.float32)
    return t, R

def _pose_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
    sd = nusc.get("sample_data", sd_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    T = Quaternion(ep["rotation"]).transformation_matrix
    T[:3, 3] = np.asarray(ep["translation"], dtype=np.float32)
    return T.astype(np.float32)

def _T_sensor_to_ego_4x4(nusc: NuScenes, sd_token: str) -> np.ndarray:
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
    throttle_max_rows: Optional[int] = 1000,
    lidar_sensor: str = DEFAULT_LIDAR,
    compute_lidar_transforms: bool = False
) -> List[Dict]:
    if cameras is None or len(cameras) == 0:
        cameras = list(DEFAULT_CAMS)
    anchor_cam = cameras[0]
    if min_future is None:
        min_future = t_out

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

        for start in range(0, len(tokens) - total + 1, stride):
            obs_tokens = tokens[start : start + t_in]
            fut_tokens = tokens[start + t_in : start + t_in + t_out]

            cams_block: Dict[str, Dict[str, list]] = {cam: {"sd_tokens": [], "img_paths": []} for cam in cameras}
            intrinsics_once: Dict[str, List[List[float]]] = {}
            obs_cam_img_grid: List[List[str]] = []

            for i, st in enumerate(obs_tokens):
                cam_info_map, ordered_paths = _cams_sd_and_imgs(nusc, st, cameras)
                obs_cam_img_grid.append(ordered_paths)
                for cam in cameras:
                    cams_block[cam]["sd_tokens"].append(cam_info_map[cam]["sd_token"])
                    cams_block[cam]["img_paths"].append(cam_info_map[cam]["img_path"])
                    if cam not in intrinsics_once:
                        intrinsics_once[cam] = cam_info_map[cam]["K"].tolist()

            lidar_sd_tokens: List[str] = []
            lidar_pc_paths: List[str] = []
            for st in obs_tokens:
                sd_l, p_l = _lidar_sd_and_path(nusc, st, lidar_sensor)
                lidar_sd_tokens.append(sd_l)
                lidar_pc_paths.append(p_l)

            T_cam_from_lidar: Dict[str, List[List[List[float]]]] = {cam: [] for cam in cameras}
            if compute_lidar_transforms:
                for i in range(len(obs_tokens)):
                    for cam in cameras:
                        cam_sd = cams_block[cam]["sd_tokens"][i]
                        lidar_sd = lidar_sd_tokens[i]
                        T = _T_cam_from_lidar_4x4(nusc, cam_sd, lidar_sd)
                        T_cam_from_lidar[cam].append(T.tolist())

            sd_anchor = cams_block[anchor_cam]["sd_tokens"][-1]
            t_w, R_w = _ego_pose_from_sd(nusc, sd_anchor)
            R_we_2x2 = R_w[:2, :2].T
            t_w_2 = t_w[:2]

            def world_xy_to_ego_xy(xy_world: Tuple[float, float]) -> List[float]:
                p = np.asarray(xy_world, dtype=np.float32) - t_w_2
                q = R_we_2x2 @ p
                return [float(q[0]), float(q[1])]

            ann_last = _ann_by_instance(nusc, obs_tokens[-1])
            if not ann_last:
                continue

            for inst_tok, last_ann in ann_last.items():
                name = last_ann.get("category_name", "")
                if not any(name.startswith(p) for p in class_prefixes):
                    continue

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

                if min_speed_mps > 0 and valid >= 2:
                    j = min(3, len(fut_xy_e) - 1)
                    x3, y3 = fut_xy_e[j]
                    dist = (x3**2 + y3**2) ** 0.5
                    approx_speed = dist / 1.5
                    if approx_speed < min_speed_mps:
                        continue

                last_xy_e = world_xy_to_ego_xy(_xy_from_ann_global(last_ann))

                if not _is_instance_in_any_cam(nusc, obs_tokens[0], inst_tok, cameras, vis=visibility):
                    continue

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
                        "sd_tokens": lidar_sd_tokens,
                        "pc_paths": lidar_pc_paths,
                        "T_cam_from_lidar": T_cam_from_lidar
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
                        "anchor_lidar_sd_token": lidar_sd_tokens[-1] if lidar_sd_tokens else None
                    },
                    "dataroot": dataroot or ""
                }

                rows.append(row)
                if len(rows) >= throttle_max_rows:
                    return rows

    return rows
