# train_traj.py
import os, sys, torch
import torch.nn as nn
import torch.optim as optim

# --- repo path (adjust if needed) ---
REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.network import XMem

# --- our modules ---
from traj.head import TrajectoryHead
from traj.predictor import XMemMMBackboneWrapper, xmem_mm_config
from traj.datamodules import NuScenesSeqLoader, collate_varK
from nuscenes.nuscenes import NuScenes


def load_xmem(backbone_ckpt="./XMem/checkpoints/XMem-s012.pth", device="cuda"):
    """
    Proper loader that reads dims from the checkpoint.
    """
    cfg = {"single_object": False}
    net = XMem(cfg, model_path=backbone_ckpt, map_location="cpu")
    state = torch.load(backbone_ckpt, map_location="cpu")  # weights_only warning is fine
    net.load_weights(state, init_as_zero_if_needed=True)
    net.to(device).eval()
    return net


def ade_fde_loss(pred_abs, gt_abs):  # [B, F, 2]
    l2 = torch.linalg.norm(pred_abs - gt_abs, dim=-1)
    ade = l2.mean()
    fde = l2[:, -1].mean()
    return ade, fde, ade + fde


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- nuScenes handle (needed to compute masks on-the-fly in Dataset) ---
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=True)

    # --- datasets / loaders (agent-expanded pickles) ---
    train_ds = NuScenesSeqLoader(
        index_path="train_agents_index.pkl",
        nusc=nusc,
        resize_hw=(360, 640),
        classes_prefix=("vehicle.", "human.pedestrian"),
        max_K=6,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=6,          # if Windows gives worker errors, set to 0 while debugging
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_varK,   # <-- keeps variable-K masks as lists
    )

    # --- XMem + memory manager backbone ---
    xmem_core = load_xmem(device=device)
    mm_cfg = xmem_mm_config(
        mem_every=3, min_mid=5, max_mid=10, num_prototypes=128,
        hidden_dim=getattr(xmem_core, "hidden_dim", 256),
        enable_long_term=False,          # keep simple/stable to start
        deep_update_every=10**9          # disable deep update at t=0
    )
    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device)

    # --- feature dim probe (one batch) ---
    first = next(iter(train_loader))
    frames0      = first["frames"].to(device)          # [B,T,3,H,W]
    init_masks0  = first["init_masks"]                 # list length B, each [K_i,H,W] (CPU)
    init_labels0 = first["init_labels"]                # list length B, each list[int]
    with torch.no_grad():
        feats0 = backbone(frames0, init_masks=init_masks0, init_labels=init_labels0)  # [B,T_feat,D]
    D = feats0.size(-1)

    # --- trajectory head ---
    head = TrajectoryHead(d_in=D, d_hid=256, horizon=train_ds.t_out).to(device)

    # freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    optim1 = optim.Adam(head.parameters(), lr=1e-3)

    # --- training loop (global XY supervision) ---
    head.train()
    for epoch in range(5):
        for batch in train_loader:
            frames      = batch["frames"].to(device)     # [B,T,3,H,W]
            init_masks  = batch["init_masks"]            # List[Tensor [K_i,H,W]] (CPU)
            init_labels = batch["init_labels"]           # List[List[int]]
            gt_future   = batch["traj"].to(device)       # [B,F,2] global XY
            last_pos    = batch["last_pos"].to(device)   # [B,2]   global XY

            # XMem features (no grad, masks moved per-sample inside wrapper)
            with torch.no_grad():
                feats = backbone(frames, init_masks=init_masks, init_labels=init_labels)  # [B,T_feat,D]

            # predict future offsets and convert to absolute (agent-centric decoding around last_pos)
            pred_offsets = head(feats)                        # [B,F,2]
            pred_abs = last_pos.unsqueeze(1) + pred_offsets.cumsum(dim=1)  # [B,F,2]

            ade, fde, loss = ade_fde_loss(pred_abs, gt_future)

            optim1.zero_grad()
            loss.backward()
            optim1.step()

        print(f"Epoch {epoch}: ADE {ade.item():.3f} | FDE {fde.item():.3f}")

if __name__ == "__main__":
    main()
