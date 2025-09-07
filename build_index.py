# build_index.py
from nuscenes.nuscenes import NuScenes
import pickle
from typing import List, Dict, Optional

NU_SCENES = r"E:\nuscenes"

def build_sequence_index(
    nusc: NuScenes,
    scene_names: Optional[List[str]],
    cam: str = "CAM_FRONT",
    t_in: int = 8,
    t_out: int = 30,
    stride: int = 1,
) -> List[Dict]:
    """
    Build an index of training sequences from nuScenes.

    Each entry in the index is a dict with keys:
      - scene_name : str
      - sample_tokens : list of str, tokens of frames in the input sequence
      - target_tokens : list of str, tokens of frames in the prediction horizon
    """
    index = []

    # Which scenes to include
    if scene_names is None:
        scenes = nusc.scene
    else:
        scenes = [s for s in nusc.scene if s["name"] in scene_names]

    for scene in scenes:
        first_sample_token = scene["first_sample_token"]
        sample = nusc.get("sample", first_sample_token)

        frames = []
        while sample is not None:
            frames.append(sample["token"])
            sample = nusc.get("sample", sample["next"]) if sample["next"] else None

        # Slide a window of length t_in + t_out over the frames
        for start in range(0, len(frames) - (t_in + t_out), stride):
            entry = {
                "scene_name": scene["name"],
                "sample_tokens": frames[start:start+t_in],
                "target_tokens": frames[start+t_in:start+t_in+t_out],
                "cam": cam,
            }
            index.append(entry)

    return index


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
