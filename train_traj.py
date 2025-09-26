import os, sys, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pickle

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.network import XMem
from xmem_mm_config import xmem_mm_config
from traj.head import MultiModalTrajectoryHead
from traj.predictor import XMemMMBackboneWrapper
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
    diff = pred_abs_k - gt_abs[:, None, :, :]
    d = torch.linalg.norm(diff, dim=-1)
    ade_k = d.mean(dim=-1)
    fde_k = torch.linalg.norm(diff[:, :, -1, :], dim=-1)
    return ade_k, fde_k


@torch.no_grad()
def metrics_best_of_k(pred_abs_k, gt_abs, r=2.0):
    ade_k, fde_k = ade_fde_per_mode(pred_abs_k, gt_abs)
    best_idx = ade_k.argmin(dim=1)
    B = gt_abs.size(0)
    row = torch.arange(B, device=gt_abs.device)
    ade = ade_k[row, best_idx].mean().item()
    fde = fde_k[row, best_idx].mean().item()
    mr = (fde_k[row, best_idx] > r).float().mean().item()
    return {"ADE": ade, "FDE": fde, "mADE": ade, "mFDE": fde, "MR@2m": mr}


import torch
import torch.nn.functional as F

def best_of_k_loss_diverse(pred_abs_k, mode_logits, gt_abs,
                           ce_weight=0.0,
                           balance_w=0.01,
                           div_w=0.05,
                           tau=4.0):
    # pred_abs_k: [B,K,T,2], gt_abs: [B,T,2]
    diff = pred_abs_k - gt_abs[:, None, :, :]
    d = torch.linalg.norm(diff, dim=-1)          # [B,K,T]
    ade_k = d.mean(dim=-1)                       # [B,K]
    fde_k = torch.linalg.norm(diff[:, :, -1, :], dim=-1)  # [B,K]

    best_idx = ade_k.argmin(dim=1)
    B, K = ade_k.shape
    row = torch.arange(B, device=gt_abs.device)

    ade_best = ade_k[row, best_idx].mean()
    fde_best = fde_k[row, best_idx].mean()
    reg = ade_best + fde_best

    ce = F.cross_entropy(mode_logits, best_idx) if ce_weight > 0.0 else pred_abs_k.new_zeros(())

    assign = F.one_hot(best_idx, num_classes=K).float()    # [B,K]
    p_bar = assign.mean(dim=0)                             # [K]
    balance = ((p_bar - (1.0 / K))**2).sum()

    flat = pred_abs_k.reshape(B, K, -1)                    # [B,K,2T]
    dist = torch.cdist(flat, flat, p=2)                    # [B,K,K]
    eye = torch.eye(K, device=gt_abs.device).unsqueeze(0)
    rep = torch.exp(-dist / tau) * (1 - eye)
    div = rep.sum(dim=(1, 2)).mean()

    total = reg + ce_weight * ce + balance_w * balance + div_w * div
    return ade_best, fde_best, total




def run_epoch(backbone, head, loader, device, optimizer=None, mr_radius=2.0, ce_weight=0.1):
    train_mode = optimizer is not None
    if train_mode:
        backbone.train(); head.train()
    else:
        backbone.eval(); head.eval()

    total = 0
    sum_ade = sum_fde = sum_made = sum_mfde = sum_mr = 0.0

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        init_masks = batch["init_masks"]
        init_labels = batch["init_labels"]
        gt_future = batch["traj"].to(device, non_blocking=True)
        last_pos = batch["last_pos"].to(device, non_blocking=True)

        feats = backbone(frames, init_masks=init_masks, init_labels=init_labels)
        traj_res_k, mode_logits = head(feats)                     # [B,K,T,2]
        pred_abs_k = last_pos[:, None, None, :] + traj_res_k

        if train_mode:
            ade, fde, loss = best_of_k_loss_diverse(
            pred_abs_k, mode_logits, gt_future,
            ce_weight=0.0, balance_w=0.01, div_w=0.05, tau=4.0
        )

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


@torch.no_grad()
def visualize_predictions(pred_abs_k, gt_abs, last_pos, save_path, max_samples=8):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    B, K, T, _ = pred_abs_k.shape
    n = min(B, max_samples)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    ade_k, _ = ade_fde_per_mode(pred_abs_k, gt_abs)
    best_idx = ade_k.argmin(dim=1).cpu().numpy()
    cmap = plt.get_cmap("tab10")

    for i in range(n):
        ax = axes[i]
        lp = last_pos[i].detach().cpu().numpy()
        gt = gt_abs[i].detach().cpu().numpy() - lp[None, :]
        preds = pred_abs_k[i].detach().cpu().numpy() - lp[None, None, :]

        ax.plot(gt[:,0], gt[:,1], linewidth=3, label="GT", color="k")

        for k in range(K):
            clr = cmap(k % 10)
            lw = 1.5 if k != best_idx[i] else 2.5
            alpha = 0.9 if k == best_idx[i] else 0.6
            ax.plot(preds[k,:,0], preds[k,:,1], linewidth=lw, alpha=alpha, label=f"M{k}" if i==0 else None, color=clr)

        ax.scatter([0.0], [0.0], s=18, color="k")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(f"Sample {i} (best=M{best_idx[i]})")

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(1+pred_abs_k.shape[1], 8), bbox_to_anchor=(0.5, 0.0))
        plt.subplots_adjust(bottom=0.12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=False)

    # Load rows from pickle
    with open("train_agents_index.pkl", "rb") as f:
        train_rows = pickle.load(f)

    with open("val_agents_index.pkl", "rb") as f:
        val_rows = pickle.load(f)

    # Create datasets
    train_ds = NuScenesSeqLoader(nusc=nusc, rows=train_rows, out_size=(384, 640))
    val_ds   = NuScenesSeqLoader(nusc=nusc, rows=val_rows,   out_size=(384, 640))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_varK)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_varK)

    xmem_core = load_xmem(device=device)

    for p in xmem_core.parameters():
        p.requires_grad = False
    for _, p in xmem_core.decoder.named_parameters():
        p.requires_grad = True
    xmem_core.decoder.train()
    xmem_core.key_encoder.eval()
    xmem_core.value_encoder.eval()

    mm_cfg = xmem_mm_config(hidden_dim=getattr(xmem_core, "hidden_dim", 256))
    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device).to(device)

    HORIZON = 10
    K = 5
    head = MultiModalTrajectoryHead(
        d_in=getattr(xmem_core, "hidden_dim", 256),
        t_out=HORIZON,
        K=K,
        hidden=256,
        dropout=0.1
    ).to(device)

    optim_all = torch.optim.AdamW([
        {"params": head.parameters(),                "lr": 1e-3, "weight_decay": 1e-4},
        {"params": xmem_core.decoder.parameters(),   "lr": 1e-5, "weight_decay": 1e-5},
    ], betas=(0.9, 0.999))

    def count_params_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print("head trainable params:", count_params_trainable(head))
    print("backbone trainable params:", count_params_trainable(backbone))
    assert any(p.requires_grad for p in head.parameters())
    assert any(p.requires_grad for p in xmem_core.decoder.parameters())

    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    os.makedirs("runs", exist_ok=True)
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

        with torch.no_grad():
            try:
                batch = next(iter(val_loader))
                frames = batch["frames"].to(device, non_blocking=True)
                init_masks = batch["init_masks"]
                init_labels = batch["init_labels"]
                gt_future = batch["traj"].to(device, non_blocking=True)
                last_pos = batch["last_pos"].to(device, non_blocking=True)
                feats = backbone(frames, init_masks=init_masks, init_labels=init_labels)
                traj_res_k, mode_logits = head(feats)                     # [B,K,T,2]
                pred_abs_k = last_pos[:, None, None, :] + traj_res_k
                visualize_predictions(pred_abs_k, gt_future, last_pos, save_path=f"runs/vis_epoch_{ep}.png", max_samples=8)
            except StopIteration:
                pass

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
