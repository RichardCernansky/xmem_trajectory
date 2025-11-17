import torch
from typing import List, Dict, Any

def collate_varK(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    def S(k): return torch.stack([b[k] for b in batch], dim=0)

    B = len(batch)
    T = batch[0]["cam_imgs"].shape[0]
    feat_dim = batch[0]["points"].shape[-1]

    maxN = 0
    for b in batch:
        for t in range(T):
            n = b["points"][t].size(0)
            if n > maxN:
                maxN = n

    points = torch.zeros((B, T, maxN, feat_dim), dtype=batch[0]["points"].dtype)
    for i, b in enumerate(batch):
        for t in range(T):
            n = b["points"][t].size(0)
            if n > 0:
                points[i, t, :n] = b["points"][t]

    init_labels =   torch.tensor([b["init_labels"] for b in batch], dtype=torch.long),

    out = {
        "cam_imgs":           S("cam_imgs"),
        "cam_depths":         S("cam_depths"),
        "cam_depth_valid":    S("cam_depth_valid"),
        "cam_K_scaled":       S("cam_K_scaled"),
        "cam_T_cam_from_ego": S("cam_T_cam_from_ego"),
        "bev_target_mask":    S("bev_target_mask"),
        "traj":               S("traj"),
        "last_pos":           S("last_pos"),
        "init_masks":         S("bev_target_mask"),
        "init_labels":        init_labels,
        "points":             points,
        "meta":               [b["meta"] for b in batch],
        "pillar_meta":        batch[0]["meta"]["pillar_meta"],
    }
    return out




