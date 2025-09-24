import os, sys, torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.network import XMem
from xmem_mm_config import xmem_mm_config
from traj.head import MultiModalTrajectoryHead           # <-- K-modal head
from traj.predictor import XMemMMBackboneWrapper        # <-- sequential RGB-only wrapper
from traj.datamodules import NuScenesSeqLoader, collate_varK
from nuscenes.nuscenes import NuScenes


def load_xmem(backbone_ckpt="./XMem/checkpoints/XMem-s012.pth", device="cuda"):
    cfg = {"single_object": False}
    net = XMem(cfg, model_path=backbone_ckpt, map_location="cpu")
    state = torch.load(backbone_ckpt, map_location="cpu")
    net.load_weights(state, init_as_zero_if_needed=True)
    net.to(device).eval()
    return net


def ade_fde_per_mode(pred_abs_k, gt_abs):
    # pred_abs_k: [B,K,T,2], gt_abs: [B,T,2]
    diff = pred_abs_k - gt_abs[:, None, :, :]              # [B,K,T,2]
    d = torch.linalg.norm(diff, dim=-1)                    # [B,K,T]
    ade_k = d.mean(dim=-1)                                 # [B,K]
    fde_k = torch.linalg.norm(diff[:, :, -1, :], dim=-1)   # [B,K]
    return ade_k, fde_k


@torch.no_grad()
def metrics_best_of_k(pred_abs_k, gt_abs, r=2.0):
    ade_k, fde_k = ade_fde_per_mode(pred_abs_k, gt_abs)    # [B,K], [B,K]
    best_idx = ade_k.argmin(dim=1)                         # [B]
    B = gt_abs.size(0)
    row = torch.arange(B, device=gt_abs.device)
    ade = ade_k[row, best_idx].mean().item()
    fde = fde_k[row, best_idx].mean().item()
    mr = (fde_k[row, best_idx] > r).float().mean().item()
    return {"ADE": ade, "FDE": fde, "mADE": ade, "mFDE": fde, "MR@2m": mr}


def best_of_k_loss(pred_abs_k, probs, gt_abs, ce_weight=0.1):
    # pred_abs_k: [B,K,T,2], probs: [B,K], gt_abs: [B,T,2]
    ade_k, fde_k = ade_fde_per_mode(pred_abs_k, gt_abs)    # [B,K], [B,K]
    best_idx = ade_k.argmin(dim=1)                         # [B]

    B = gt_abs.size(0)
    row = torch.arange(B, device=gt_abs.device)
    min_ade = ade_k[row, best_idx].mean()
    min_fde = fde_k[row, best_idx].mean()

    # cross-entropy to push probabilities toward the best mode
    ce = torch.nn.functional.cross_entropy(probs, best_idx)

    return min_ade, min_fde, (min_ade + min_fde) + ce_weight * ce


def run_epoch(backbone, head, loader, device, optimizer=None, mr_radius=2.0, ce_weight=0.1):
    train_mode = optimizer is not None
    if train_mode:
        backbone.train(); head.train()
    else:
        backbone.eval(); head.eval()

    total = 0
    sum_ade = sum_fde = sum_made = sum_mfde = sum_mr = 0.0

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)     # [B,T,3,H,W]
        init_masks = batch["init_masks"]                           # len B
        init_labels = batch["init_labels"]                         # len B
        gt_future = batch["traj"].to(device, non_blocking=True)    # [B,T,2]
        last_pos = batch["last_pos"].to(device, non_blocking=True) # [B,2]

        # features from sequential RGB backbone
        feats = backbone(frames, init_masks=init_masks, init_labels=init_labels)  # [B,Tf,D]

        # multimodal head -> offsets + mode probs
        traj_offsets, mode_probs = head(feats)                     # [B,K,T,2], [B,K]

        # turn offsets into absolute positions
        pred_abs_k = last_pos[:, None, None, :] + traj_offsets.cumsum(dim=2)  # [B,K,T,2]

        if train_mode:
            ade, fde, loss = best_of_k_loss(pred_abs_k, mode_probs, gt_future, ce_weight=ce_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            m = {"ADE": ade.item(), "FDE": fde.item(),
                 "mADE": ade.item(), "mFDE": fde.item(),
                 "MR@2m": metrics_best_of_k(pred_abs_k, gt_future, r=mr_radius)["MR@2m"]}
        else:
            with torch.no_grad():
                m = metrics_best_of_k(pred_abs_k, gt_future, r=mr_radius)

        bsz = gt_future.shape[0]
        sum_ade += m["ADE"] * bsz
        sum_fde += m["FDE"] * bsz
        sum_made += m["mADE"] * bsz
        sum_mfde += m["mFDE"] * bsz
        sum_mr += m["MR@2m"] * bsz
        total += bsz

    return {
        "ADE": sum_ade / max(1, total),
        "FDE": sum_fde / max(1, total),
        "mADE": sum_made / max(1, total),
        "mFDE": sum_mfde / max(1, total),
        "MR@2m": sum_mr / max(1, total),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=False)

    train_ds = NuScenesSeqLoader(index_path="train_agents_index.pkl", resize=True, resize_wh=(640, 384), nusc=nusc)
    val_ds   = NuScenesSeqLoader(index_path="val_agents_index.pkl",   resize=True, resize_wh=(640, 384), nusc=nusc)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_varK)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_varK)

    xmem_core = load_xmem(device=device)

    # --- Freeze, then unfreeze only decoder initially ---
    for p in xmem_core.parameters():
        p.requires_grad = False
    for _, p in xmem_core.decoder.named_parameters():
        p.requires_grad = True
    xmem_core.decoder.train()
    xmem_core.key_encoder.eval()
    xmem_core.value_encoder.eval()

    mm_cfg = xmem_mm_config(hidden_dim=getattr(xmem_core, "hidden_dim", 256))
    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device).to(device)

    # ---- K-modal head settings ----
    HORIZON = 10
    K = 5
    head = MultiModalTrajectoryHead(
        d_in=getattr(xmem_core, "hidden_dim", 256),
        t_out=HORIZON,
        K=K,
        hidden=256,
        dropout=0.1
    ).to(device)

    # Optimizer (no fusion params in RGB-only backbone)
    optim_all = torch.optim.AdamW([
        {"params": head.parameters(),                "lr": 1e-3, "weight_decay": 1e-4},
        {"params": xmem_core.decoder.parameters(),   "lr": 1e-5, "weight_decay": 1e-5},
    ], betas=(0.9, 0.999))

    # quick checks
    def count_params_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("head trainable params:", count_params_trainable(head))
    print("backbone trainable params:", count_params_trainable(backbone))
    assert any(p.requires_grad for p in head.parameters()), "No trainable params in head!"
    assert any(p.requires_grad for p in xmem_core.decoder.parameters()), "Decoder not trainable!"

    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    EPOCHS = 20
    for ep in range(EPOCHS):
        train_m = run_epoch(backbone, head, train_loader, device, optimizer=optim_all, mr_radius=2.0, ce_weight=0.1)
        val_m   = run_epoch(backbone, head, val_loader,   device, optimizer=None,    mr_radius=2.0)
        print(f"Epoch {ep}: Train ADE {train_m['ADE']:.3f} FDE {train_m['FDE']:.3f} | "
              f"Val ADE {val_m['ADE']:.3f} FDE {val_m['FDE']:.3f} MR@2m {val_m['MR@2m']:.3f}")

        hist["epoch"].append(ep)
        hist["train_ADE"].append(train_m["ADE"])
        hist["train_FDE"].append(train_m["FDE"])
        hist["val_ADE"].append(val_m["ADE"])
        hist["val_FDE"].append(val_m["FDE"])
        hist["val_MR2"].append(val_m["MR@2m"])

    os.makedirs("runs", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(hist["epoch"], hist["train_ADE"], marker="o", label="Train ADE")
    plt.plot(hist["epoch"], hist["val_ADE"], marker="o", label="Val ADE")
    plt.plot(hist["epoch"], hist["train_FDE"], marker="o", label="Train FDE")
    plt.plot(hist["epoch"], hist["val_FDE"], marker="o", label="Val FDE")
    plt.xlabel("Epoch"); plt.ylabel("Error"); plt.title("ADE/FDE"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("runs/ade_fde.png")

    plt.figure(figsize=(8,4))
    plt.plot(hist["epoch"], hist["val_MR2"], marker="o", label="Val MR@2m")
    plt.xlabel("Epoch"); plt.ylabel("Miss rate"); plt.title("Miss Rate @ 2 m"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("runs/missrate.png")


if __name__ == "__main__":
    main()
