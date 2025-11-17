import torch
from typing import List, Dict, Any

def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    def S(k): return torch.stack([b[k] for b in batch], dim=0)

    B = len(batch)
    T = batch[0]["pillar_features"].shape[0]
    M = batch[0]["pillar_features"].shape[2]
    C_feat = batch[0]["pillar_features"].shape[3]
    P_max = max(b["pillar_features"].shape[1] for b in batch)

    pillar_features = torch.zeros((B, T, P_max, M, C_feat), dtype=batch[0]["pillar_features"].dtype)
    pillar_coords   = torch.zeros((B, T, P_max, 2), dtype=torch.int32)
    pillar_npoints  = torch.zeros((B, T, P_max), dtype=torch.int32)
    for i, b in enumerate(batch):
        P_i = b["pillar_features"].shape[1]
        if P_i > 0:
            pillar_features[i, :, :P_i] = b["pillar_features"]
            pillar_coords[i, :, :P_i]   = b["pillar_coords"]
            pillar_npoints[i, :, :P_i]  = b["pillar_num_points"]

    out = {
        "cam_imgs":            S("cam_imgs"),
        "cam_depths":          S("cam_depths"),
        "cam_depth_valid":     S("cam_depth_valid"),
        "cam_K_scaled":        S("cam_K_scaled"),
        "cam_T_cam_from_ego":  S("cam_T_cam_from_ego"),
        "bev_target_mask":     S("bev_target_mask"),
        "traj":                S("traj"),
        "last_pos":            S("last_pos"),
        "init_masks":          S("bev_target_mask"),
        "init_labels":         torch.tensor([b["init_labels"] for b in batch], dtype=torch.long),
        "meta":                [b["meta"] for b in batch],
        "pillar_features":     pillar_features,
        "pillar_coords":       pillar_coords,
        "pillar_num_points":   pillar_npoints,
        "pillar_meta":         batch[0]["pillar_meta"],
    }
    return out
