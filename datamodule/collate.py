import torch
from typing import List, Dict, Any

def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # just stack along batch; no pano assembly here
    def S(k): return torch.stack([b[k] for b in batch], dim=0)

    out = {
        "cam_imgs":            S("cam_imgs"),            # (B,T,C,3,H,cw)
        "cam_depths":          S("cam_depths"),          # (B,T,C,H,cw)
        "cam_depth_valid":     S("cam_depth_valid"),     # (B,T,C,H,cw)
        "cam_K_scaled":        S("cam_K_scaled"),        # (B,T,C,3,3)
        "cam_T_cam_from_ego":  S("cam_T_cam_from_ego"),  # (B,T,C,4,4)
        "lidar_bev_raw":       S("lidar_bev_raw"),       # (B,T,4,Hb,Wb)
        "bev_target_mask":     S("bev_target_mask"),     # (B,T,1,Hb,Wb)
        "traj":                S("traj"),                # (B,T_out,2)
        "last_pos":            S("last_pos"),            # (B,2)
        "init_masks":          S("bev_target_mask"),
        "init_labels": torch.tensor([b["init_labels"] for b in batch], dtype=torch.long),  # (B,)
        "meta":                [b["meta"] for b in batch]
    }
    return out
