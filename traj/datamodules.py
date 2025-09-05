import torch
from torch.utils.data import Dataset
# You’ll use the official nuScenes devkit to fetch frames + annotations

class NuScenesSeqLoader(Dataset):
    def __init__(self, split="train", t_in=8, t_out=30, modality="rgb"):
        self.t_in, self.t_out = t_in, t_out
        self.modality = modality
        # TODO: index your sequences here via nuScenes API

    def __len__(self): 
        return 1000  # placeholder

    def __getitem__(self, idx):
        # 1) create T_in frames, each [C,H,W]
        frames = torch.stack([torch.randn(3,360,640) for _ in range(self.t_in)], dim=0)  # [T,C,H,W]

        traj = torch.randn(self.t_out, 2)   # [T_out,2]
        last_pos = torch.randn(2)           # [2]

        return {"frames": frames, "traj": traj, "last_pos": last_pos}
