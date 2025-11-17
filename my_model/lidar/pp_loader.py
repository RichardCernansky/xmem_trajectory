import torch
from typing import Dict, Any, Tuple, Optional
from my_model.lidar.pp_encoder import PointPillarsEncoder

def build_pointpillars(cfg: Dict[str, Any], device: str):
    enc = PointPillarsEncoder(
        pts_voxel_encoder=cfg["pts_voxel_encoder"],
        pts_middle_encoder=cfg["pts_middle_encoder"],
        pts_backbone=cfg.get("pts_backbone"),
        pts_neck=cfg.get("pts_neck"),
        pp_head=cfg.get("pp_head"),
        max_num_points=cfg.get("max_num_points", 10),
        max_voxels=cfg.get("max_voxels", [90000, 120000]),
    ).to(device)
    return enc

def load_pp_backbone_weights(enc: PointPillarsEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    keep = ("pts_voxel_encoder", "pts_middle_encoder", "pts_backbone", "pts_neck")
    sd_backbone = {k: v for k, v in sd.items() if k.startswith(keep)}
    missing, unexpected = enc.load_state_dict(sd_backbone, strict=False)
    return missing, unexpected
