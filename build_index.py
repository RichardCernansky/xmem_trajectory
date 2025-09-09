# build_index_agents.py
import pickle
from nuscenes.nuscenes import NuScenes
from agent_index import build_agent_sequence_index

nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=True)

common = dict(
    cam="CAM_FRONT",
    t_in=8,
    t_out=30,
    stride=1,
    min_future=30,                  # require >= 20 future steps
    min_speed_mps=0.1,              # skip near-stationary targets
    class_prefixes=("vehicle.", "human.pedestrian"),
)

train_rows = build_agent_sequence_index(nusc, splits="train", **common)
print(f"train rows: {len(train_rows)}")
with open("train_agents_index.pkl", "wb") as f:
    pickle.dump(train_rows, f)

val_rows = build_agent_sequence_index(nusc, splits="val", **common)
print(f"val rows: {len(val_rows)}")
with open("val_agents_index.pkl", "wb") as f:
    pickle.dump(val_rows, f)
