import torch

def collate_varK(batch):
    # RGB pano: (B, T_in, 3, H_pano, W_pano)
    frames = torch.stack([b["frames"] for b in batch], dim=0)

    # Depth pano + extras: (B, T_in, 1/2, H_pano, W_pano)
    depths = torch.stack([b["depths"] for b in batch], dim=0)
    depth_extras = torch.stack([b["depth_extras"] for b in batch], dim=0)

    # LiDAR BEV (raw stats): (B, T_in, 4, H_raw, W_raw)
    lidar_bev_raw = torch.stack([b["lidar_bev_raw"] for b in batch], dim=0)

    # Supervision
    traj = torch.stack([b["traj"] for b in batch], dim=0)     # (B, T_out, 2)
    last = torch.stack([b["last_pos"] for b in batch], dim=0) # (B, 2)

    # Initial mask & labels
    init_masks  = torch.stack([b["init_masks"] for b in batch], dim=0)  # (B, 1, H_pano, W_pano) uint8
    # collapse inner list like [1] -> 1 to get shape (B,)
    init_labels = torch.tensor([b["init_labels"] for b in batch], dtype=torch.long)  # (B,)

    # Metadata stays a list of dicts
    meta = [b["meta"] for b in batch]

    return {
        "frames": frames,
        "depths": depths,
        "depth_extras": depth_extras,
        "lidar_bev_raw": lidar_bev_raw,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }






