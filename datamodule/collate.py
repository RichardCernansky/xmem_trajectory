import torch

def collate_varK(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    traj   = torch.stack([b["traj"] for b in batch], dim=0)
    last   = torch.stack([b["last_pos"] for b in batch], dim=0)
    init_masks  = [b["init_masks"] for b in batch]
    init_labels = [b["init_labels"] for b in batch]
    meta = [b["meta"] for b in batch]
    return {
        "frames": frames,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
    }