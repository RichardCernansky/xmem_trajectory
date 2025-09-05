Folder structure:
XMem/
 ├─ xmem/                # original
 ├─ traj/                # <-- new
 │   ├─ predictor.py     # wrapper that uses XMem encoders+memory
 │   ├─ head.py          # small GRU/MLP head
 │   └─ datamodules.py   # lightweight nuScenes sequence loader (stub)
 └─ train_traj.py        # your training script

# xmem_trajectory
