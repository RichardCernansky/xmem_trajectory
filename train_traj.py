# train_traj.py
import os, sys, torch
import torch.nn as nn
import torch.optim as optim

from traj.head import TrajectoryHead
from traj.datamodules import NuScenesSeqLoader   # your dataset class

# --- repo path (adjust if needed) ---
REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.network import XMem

# --- memory-manager backbone wrapper ---
from traj.predictor import XMemMMBackboneWrapper, xmem_mm_config

def load_xmem(backbone_ckpt="./XMem/checkpoints/XMem-s012.pth", device="cuda"):
    """
    Proper loader that lets XMem read dims from the checkpoint.
    """
    cfg = {"single_object": False}
    net = XMem(cfg, model_path=backbone_ckpt, map_location="cpu")
    state = torch.load(backbone_ckpt, map_location="cpu")
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

    # --- data ---
    # Your loader should return:
    #   batch["frames"] : list length T_in of [B,3,H,W] tensors (already normalized)
    #   batch["traj"]   : [B, F, 2] future absolute positions
    #   batch["last_pos"]: [B, 2] last observed absolute position (t = T_in-1)
    train_ds = NuScenesSeqLoader(split="train", t_in=8, t_out=30, modality="rgb")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    # --- XMem + memory manager backbone ---
    xmem_core = load_xmem(device=device)

    mm_cfg = xmem_mm_config(mem_every=3, min_mid=5, max_mid=10, num_prototypes=128, hidden_dim=getattr(xmem_core, "hidden_dim", 256))
    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device)

    # --- Trajectory head ---
    # Do a tiny dry run to get D (feature dimension)
    # Use a single batch from loader
    first = next(iter(train_loader))
    frames0 = first["frames"].to(device)   # stays [B,T,C,H,W]
    with torch.no_grad():
        feats0 = backbone(frames0)         # [B, T_feat, D]
    D = feats0.size(-1)

    head = TrajectoryHead(d_in=D, d_hid=256, horizon=train_ds.t_out).to(device)

    # Freeze backbone (it already runs no_grad internally)
    for p in backbone.parameters():
        p.requires_grad = False

    optim1 = optim.Adam(head.parameters(), lr=1e-3)

    # --- Train head only ---
    head.train()
    for epoch in range(5):
        for batch in train_loader:
            # move frames (list!) to device
            frames = batch["frames"].to(device)  # stays [B,T,C,H,W]
            gt_future = batch["traj"].to(device)       # [B, F, 2]
            last_pos  = batch["last_pos"].to(device)   # [B, 2]

            with torch.no_grad():
                feats_seq = backbone(frames)           # [B, T_feat, D]

            pred_offsets = head(feats_seq)             # [B, F, 2]
            # convert offsets → absolute
            pred_abs = last_pos.unsqueeze(1) + pred_offsets.cumsum(dim=1)

            ade, fde, loss = ade_fde_loss(pred_abs, gt_future)
            optim1.zero_grad()
            loss.backward()
            optim1.step()

        print(f"Epoch {epoch}: ADE {ade.item():.3f}  FDE {fde.item():.3f}")

    # Optional: fine-tuning XMem is non-trivial with InferenceCore (step() usually runs no_grad).
    # Stick to head-only training unless you rework the wrapper to allow grads.

if __name__ == "__main__":
    main()
