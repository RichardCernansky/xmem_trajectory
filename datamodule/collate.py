import torch

def collate_varK(batch):
    # RGB pano: (B, T_in, 3, H, W)
    frames = torch.stack([b["frames"] for b in batch], dim=0)

    # Depth pano: (B, T_in, 1, H, W)
    depths = torch.stack([b["depths"] for b in batch], dim=0)
    depth_extras = torch.stack([b["depth_extras"] for b in batch], dim=0)

    # Supervision
    traj = torch.stack([b["traj"] for b in batch], dim=0)          # (B, T_future, 2)
    last = torch.stack([b["last_pos"] for b in batch], dim=0)      # (B, 2)

    # Initial mask & labels
    init_masks  = torch.stack([b["init_masks"] for b in batch], dim=0)  # (B, 1, H, W) uint8
    init_labels = torch.tensor([b["init_labels"] for b in batch], dtype=torch.long)  # (B,)

    # Metadata stays a list of dicts
    meta = [b["meta"] for b in batch]

    return {
        "frames": frames,
        "depths": depths,
        "depth_extras": depth_extras,  
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }
