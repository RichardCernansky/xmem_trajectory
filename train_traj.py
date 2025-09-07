# train_traj.py
import os, sys, torch
import torch.nn as nn
import torch.optim as optim

from traj.head import TrajectoryHead
from traj.datamodules import NuScenesSeqLoader   # your dataset class

from nuscenes.nuscenes import NuScenes
from traj.datamodules import NuScenesSeqLoader, collate_varK
# --- memory-manager backbone wrapper ---
from traj.predictor import XMemMMBackboneWrapper, xmem_mm_config
from model.network import XMem


# --- repo path (adjust if needed) ---
REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NU_SCENES = r"E:\nuscenes"

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

    # --- nuScenes handle (needed for on-the-fly masks) ---
    # Make sure 'dataroot' points to your local nuScenes folder
    nusc = NuScenes(version="v1.0-trainval", dataroot=NU_SCENES, verbose=True)

    # --- Dataset from prebuilt index + on-the-fly t=0 masks ---
    train_ds = NuScenesSeqLoader(
        index_path="train_index.pkl",   # produced by build_index.py
        nusc=nusc,
        resize_hw=(360, 640),
        classes_prefix=("vehicle.car", "vehicle.truck", "vehicle.bus"),
        max_K=4,
    )

    # --- DataLoader (custom collate for variable-K masks) ---
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_varK,   # <- critical for variable-K masks
    )

    # --- XMem + memory manager backbone ---
    xmem_core = load_xmem(device=device)
    mm_cfg = xmem_mm_config(
        mem_every=3,
        min_mid=5,
        max_mid=10,
        num_prototypes=128,
        hidden_dim=getattr(xmem_core, "hidden_dim", 256),
        enable_long_term=False,
        deep_update_every=10**9,   # keep deep-update off to avoid channel mismatch
        single_object=False,
    )
    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device)

    # --- Trajectory head dimension via a tiny dry run ---
    first = next(iter(train_loader))
    frames0      = first["frames"].to(device)          # [B,T,3,H,W]
    init_masks0  = first["init_masks"]                 # list of [K_i,H,W] (CPU)
    init_labels0 = first["init_labels"]                # list[list[int]]

    with torch.no_grad():
        feats0 = backbone(frames0, init_masks=init_masks0, init_labels=init_labels0)  # [B,T_feat,D]
    D = feats0.size(-1)

    head = TrajectoryHead(d_in=D, d_hid=256, horizon=first["traj"].size(1)).to(device)

    # Freeze backbone (feature extractor only)
    for p in backbone.parameters():
        p.requires_grad = False

    optim1 = torch.optim.Adam(head.parameters(), lr=1e-3)

    # --- Train head only ---
    head.train()
    for epoch in range(5):
        for batch in train_loader:
            frames      = batch["frames"].to(device)        # [B, T, 3, H, W]
            init_masks  = batch["init_masks"]               # list of [K_i, H, W] (CPU; leave as-is)
            init_labels = batch["init_labels"]              # list[list[int]]
            gt_future   = batch["traj"].to(device)          # [B, F, 2]
            last_pos    = batch["last_pos"].to(device)      # [B, 2]

            with torch.no_grad():
                feats = backbone(                           # [B, T_feat, D]
                    frames,
                    init_masks=init_masks,
                    init_labels=init_labels
                )

            pred_offsets = head(feats)                      # [B, F, 2]
            pred_abs = last_pos.unsqueeze(1) + pred_offsets.cumsum(dim=1)  # [B, F, 2]

            ade, fde, loss = ade_fde_loss(pred_abs, gt_future)

            optim1.zero_grad()
            loss.backward()
            optim1.step()

        print(f"Epoch {epoch}: ADE {ade.item():.3f}  FDE {fde.item():.3f}")

    # Note: fine-tuning XMem is non-trivial (InferenceCore.step runs no_grad);
    # stick to head-only training unless you rework the wrapper to allow grads.


if __name__ == "__main__":
    main()
