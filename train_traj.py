import torch, torch.nn as nn, torch.optim as optim
from traj.head import TrajectoryHead
from traj.predictor import XMemBackboneWrapper
from traj.datamodules import NuScenesSeqLoader   # you’ll write a minimal version

import os, sys
repo_root = os.path.join(os.path.dirname(__file__), "XMem")  # or "xmem" if you renamed it
sys.path.insert(0, repo_root)
from model import network  # OR: from model import XMem if you export it; see below

def get_config(
    mem_every=3,            # r in the paper: write interval (↑ speed, ↓ detail)
    deep_update_every=-1,   # -1 means "sync with mem_every"
    enable_long_term=True,  # use long-term (prototype) memory
    min_mid=5, max_mid=10,  # T_min / T_max for mid-term buffer
    num_prototypes=128,     # P in the paper
    max_long_term=10000,    # LT_max (global cap on long-term elems)
    top_k=30,               # top-K memory matches (speed/VRAM trade-off)
    benchmark=False         # leave False unless you care about FPS timing
):
    cfg = {
        # --- long-term memory flags/limits ---
        "enable_long_term": enable_long_term,
        "max_mid_term_frames": max_mid,     # T_max
        "min_mid_term_frames": min_mid,     # T_min
        "max_long_term_elements": max_long_term,  # LT_max
        "num_prototypes": num_prototypes,   # P

        # --- write cadence / matching ---
        "mem_every": mem_every,             # r
        "deep_update_every": deep_update_every,
        "top_k": top_k,

        # optional helpers used in the official scripts
        "benchmark": benchmark,
        # You can leave this False; set True only for very long videos to enable usage-based pruning logic
        "enable_long_term_count_usage": False,
    }
    return cfg

def load_xmem(backbone_ckpt="./XMem/checkpoints/XMem-s012.pth", device="cuda"):
    # minimal config (will be updated by init_hyperparameters inside XMem)
    cfg = get_config()
    # construct the model
    model = network.XMem(cfg, model_path=backbone_ckpt, map_location="cpu")
    # load weights properly
    state = torch.load(backbone_ckpt, map_location="cpu")
    model.load_weights(state, init_as_zero_if_needed=True)

    return model.to(device).eval()

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def ade_fde_loss(pred, gt):  # pred/gt: [B, T_out, 2]
    l2 = torch.linalg.norm(pred - gt, dim=-1)  # [B,T]
    ade = l2.mean()
    fde = l2[:, -1].mean()
    return ade, fde, ade + fde


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # --- data ---
    train_ds = NuScenesSeqLoader(split="train", t_in=8, t_out=30, modality="rgb")  # or "bev"
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)

    # --- backbone + head ---
    xmem_core = load_xmem()
    backbone = XMemBackboneWrapper(xmem_core, write_interval=3, use_roi=False).to(device)

    # Stage 1: freeze encoders+memory
    freeze(backbone.xmem)

    head = TrajectoryHead(d_in=backbone.feat_dim, d_hid=256, horizon=30).to(device)

    # Only head is trainable in stage 1
    optim1 = optim.Adam(head.parameters(), lr=1e-3)

    # ----- Stage 1 training -----
    head.train(); backbone.eval()
    for epoch in range(5):  # warmup epochs; tune as needed
        for batch in train_loader:
            frames, gt_traj = batch["frames"].to(device), batch["traj"].to(device)   # frames: list(T_in) of [B,C,H,W]
            feats_seq = backbone(frames)                                             # [B,T_in,D]
            pred = head(feats_seq)                                                   # [B,T_out,2]
            ade, fde, loss = ade_fde_loss(pred, gt_traj)

            optim1.zero_grad()
            loss.backward()
            optim1.step()

    # ----- Stage 2 fine-tune encoders (optional) -----
    for p in backbone.xmem.parameters(): p.requires_grad = True
    optim2 = optim.Adam([
        {"params": backbone.xmem.parameters(), "lr": 1e-4},
        {"params": head.parameters(),           "lr": 5e-4},
    ])

    head.train(); backbone.train()
    for epoch in range(20):
        for batch in train_loader:
            frames, gt_traj = batch["frames"].to(device), batch["traj"].to(device)
            feats_seq = backbone(frames)
            pred = head(feats_seq)
            ade, fde, loss = ade_fde_loss(pred, gt_traj)

            optim2.zero_grad()
            loss.backward()
            optim2.step()

if __name__ == "__main__":
    main()
