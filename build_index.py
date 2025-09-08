# build_index.py
from nuscenes.nuscenes import NuScenes
import pickle
from typing import List, Dict, Optional
import numpy as np

NU_SCENES = r"E:\nuscenes"

def build_sequence_index(
    nusc: NuScenes,
    scene_names: Optional[List[str]] = None,   # None = use all scenes
    cam: str = "CAM_FRONT",
    t_in: int = 8,
    t_out: int = 30,
    stride: int = 1,                           # slide window by this many samples
) -> List[Dict]:
    """
    Build a flat list of training windows (each a dict) with:
      scene_name, start_sample_token, obs_sample_tokens, fut_sample_tokens,
      cam_sd_tokens (for obs frames), img_paths (for obs frames), intrinsics (for obs frames), cam
    """
    windows: List[Dict] = []

    # choose scenes
    scenes = (
        nusc.scene if scene_names is None
        else [s for s in nusc.scene if s["name"] in set(scene_names)]
    )

    for sc in scenes:
        scene_name = sc["name"]

        # collect all sample tokens (timeline) for this scene
        tokens = []
        tok = sc["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            tokens.append(tok)
            tok = s["next"] if s["next"] != "" else None

        total = t_in + t_out
        if len(tokens) < total:
            continue  # scene too short

        # slide a window over the scene timeline
        for start in range(0, len(tokens) - total + 1, stride):
            obs_tokens = tokens[start : start + t_in]
            fut_tokens = tokens[start + t_in : start + total]

            # resolve camera sample_data + paths + intrinsics for observed frames
            cam_sd_tokens, img_paths, intrinsics = [], [], []
            for st in obs_tokens:
                s = nusc.get("sample", st)
                sd_tok = s["data"][cam]                         # camera sample_data token
                cam_sd_tokens.append(sd_tok)
                img_path, _, K = nusc.get_sample_data(sd_tok, box_vis_level=0)
                img_paths.append(img_path)                      # absolute path
                intrinsics.append(np.asarray(K, dtype=np.float32).tolist())  # 3x3 → list for pickle

            windows.append({
                "scene_name": scene_name,
                "start_sample_token": obs_tokens[0],
                "obs_sample_tokens": obs_tokens,       # length = t_in
                "fut_sample_tokens": fut_tokens,       # length = t_out
                "cam_sd_tokens": cam_sd_tokens,        # len = t_in (obs only)
                "img_paths": img_paths,                # len = t_in (obs only)
                "intrinsics": intrinsics,              # len = t_in, each a 3x3 list
                "cam": cam,
            })

    return windows



# Initialize nuScenes (point dataroot to where you unpacked nuScenes)
# version="v1.0-trainval" = main 700-scene split
nusc = NuScenes(version="v1.0-trainval", dataroot=NU_SCENES, verbose=True)

# Define splits you want to build indexes for.
# Replace None with a list of scene names if you want a manual split.
splits = {
    "train": None,   # use all scenes for training (or filter list of scenes)
    "val": None,     # use all scenes for validation (or filter list of scenes)
}

for split_name, scene_names in splits.items():
    index = build_sequence_index(
        nusc=nusc,
        scene_names=scene_names,  # list of scene names to include, or None for all
        cam="CAM_FRONT",          # which camera stream to use (e.g. "CAM_FRONT", "CAM_BACK")
        t_in=8,                   # number of observed frames for the model input
        t_out=30,                 # number of future frames for the trajectory prediction
        stride=1,                 # sliding window stride (1 = every frame, 2 = skip every other, etc.)
    )

    print(f"{split_name} has {len(index)} sequences")

    # Save the index as a pickle for fast reloading in Dataset
    with open(f"{split_name}_index.pkl", "wb") as f:
        pickle.dump(index, f)
