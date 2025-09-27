import os, sys, torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pickle

@torch.no_grad()
def visualize_xmem_masks_simple(backbone, frames, save_path, max_samples=4, max_frames=6, thresh=0.5):
    import numpy as np, matplotlib.pyplot as plt
    # Normalize frames to [B,T,3,H,W]
    if frames.ndim == 5 and frames.shape[1] == 3:
        frames = frames.permute(0,2,1,3,4)
    elif frames.ndim == 4:
        frames = frames.unsqueeze(0)
    F_btchw = frames.detach().cpu()
    masks = getattr(backbone, "last_masks", None)  # [B,T,H,W]
    if masks is None:
        print("[mask-viz] backbone.last_masks is None — run backbone(...) first.")
        return
    B, T, H, W = masks.shape
    nB, nT = min(B, max_samples), min(T, max_frames)
    fig, axes = plt.subplots(nB, nT, figsize=(3.2*nT, 3.0*nB))
    if nB == 1 and nT == 1: axes = np.array([[axes]])
    elif nB == 1:           axes = axes[None, :]
    elif nT == 1:           axes = axes[:, None]
    for i in range(nB):
        for t in range(nT):
            ax = axes[i, t]
            img = F_btchw[i, t].permute(1,2,0).numpy()
            img = (img*255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            m = (masks[i, t].numpy() >= thresh).astype(np.uint8)
            ax.imshow(img); ax.imshow(m, alpha=0.45, cmap="Reds")
            ax.set_axis_off()
            if i == 0: ax.set_title(f"t={t}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140); plt.close(fig)



# --- time-ramp constants (shared prefix control) ---
ENABLE_BRANCH_RAMP = False
RAMP_L0 = 2
RAMP_L1 = 1
RAMP_SHARP0 = 10.0
RAMP_SHARP1 = 12.0

WARMUP_NO_RAMP_EPOCHS = 2  # epochs 0..1 = no ramp
RELAX_FRACTION = 0.8       # relax in the last 20% of training

# ---- yaw alignment (LOSS-ONLY) ----
ENABLE_YAW_ALIGN = True      # set True to enable
YAW_MIN_SPD = 0.15           # m/step; if first-step speed < this, skip rotation

# ---- soft prefix knobs ----
L_PREFIX = 3              # first L steps encouraged to match
TW_ALPHA = 1.0              # time-weight decay (smaller = heavier early steps)
W_PREFIX_VAR_MAX  = 0.03    # final weight for prefix variance
W_HEAD_ALIGN_MAX  = 0.05    # final weight for heading alignment
SOFT_WARMUP_EPOCHS = 3     # ramp the two weights from 0 -> max over these epochs

def yaw_align_for_loss(pred_abs_k, gt_abs, last_pos, min_spd=YAW_MIN_SPD):
    rel_gt   = gt_abs  - last_pos[:, None, :]              # [B,T,2]
    rel_pred = pred_abs_k - last_pos[:, None, None, :]     # [B,K,T,2]

    disp = torch.norm(rel_gt, dim=-1)                      # [B,T]
    B, T = disp.shape
    has  = disp > min_spd
    t_idx = torch.where(
        has.any(dim=1),
        has.float().argmax(dim=1),                         # first index with motion
        torch.ones(B, device=rel_gt.device, dtype=torch.long).clamp_max(T-1)  # fallback=1
    )
    v0  = rel_gt[torch.arange(B, device=rel_gt.device), t_idx]  # [B,2]
    ang = torch.atan2(v0[:,1], v0[:,0]); c, s = torch.cos(ang), torch.sin(ang)
    R = torch.stack([torch.stack([ c,  s], -1),
                     torch.stack([-s,  c], -1)], -2)       # [B,2,2]

    gt_rel_r   = torch.einsum('bti,bij->btj',   rel_gt,   R)
    pred_rel_r = torch.einsum('bkti,bij->bktj', rel_pred, R)
    return pred_rel_r, gt_rel_r

def ramp_schedule(epoch: int, total_epochs: int):
    # Returns (L, sharp) for this epoch.
    # Caller should still gate with: use_ramp_now = ENABLE_BRANCH_RAMP and (epoch >= WARMUP_NO_RAMP_EPOCHS)
    if epoch < WARMUP_NO_RAMP_EPOCHS:
        # Value won't be used because use_ramp_now will be False
        return 0, 0.0

    s = epoch / max(1, total_epochs - 1)
    if s < 0.25:
        L, sharp = 1, 8.0     # light tie while things start moving
    elif s < RELAX_FRACTION:
        L, sharp = 2, 12.0    # enforce a clean shared prefix
    else:
        L, sharp = 1, 14.0    # relax late for crisp branching
    return L, sharp

def apply_branch_ramp(traj_res_k, L: int, sharp: float):
    # Hard tie for the first L steps, smooth release afterwards.
    B, K, T, _ = traj_res_k.shape
    t = torch.arange(T, device=traj_res_k.device, dtype=traj_res_k.dtype)
    w = torch.sigmoid((t - L) * sharp).view(1, 1, T, 1)  # ~0 before L, ~1 after
    if L > 0:
        w[..., :L, :] = 0.0  # EXACT tie for first L steps
    mean = traj_res_k.mean(dim=1, keepdim=True)
    return (1 - w) * mean + w * traj_res_k


REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.network import XMem
from xmem_mm_config import xmem_mm_config
from traj.head import MultiModalTrajectoryHead
from traj.predictor import XMemMMBackboneWrapper
from traj.datamodules import NuScenesSeqLoader, collate_varK
from nuscenes.nuscenes import NuScenes
import torch.nn.functional as F


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


def soft_cross_entropy(logits: torch.Tensor,
                       target_probs: torch.Tensor,
                       *,
                       detach_targets: bool = True,
                       label_smoothing: float = 0.0,
                       eps: float = 1e-8) -> torch.Tensor:
    """
    logits: [B,K] raw scores from classifier
    target_probs: [B,K] soft targets (e.g., aWTA weights)

    - Detaches targets by default (recommended for gating)
    - Clamps/renormalizes to avoid log(0)
    - Optional label smoothing (tiny) for extra stability
    """
    if detach_targets:
        target_probs = target_probs.detach()

    # align dtype/device
    target_probs = target_probs.to(dtype=logits.dtype, device=logits.device)

    # clamp + renorm → valid distribution
    target_probs = target_probs.clamp_min(eps)
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)

    # optional tiny smoothing
    if label_smoothing > 0.0:
        K = target_probs.size(1)
        uni = target_probs.new_full((1, K), 1.0 / K)
        target_probs = (1.0 - label_smoothing) * target_probs + label_smoothing * uni

    logp = F.log_softmax(logits, dim=1)
    loss = -(target_probs * logp).sum(dim=1).mean()
    return loss

def soft_weights_for_epoch(epoch):
    s = min(1.0, float(epoch) / max(1, SOFT_WARMUP_EPOCHS))
    return W_PREFIX_VAR_MAX * s, W_HEAD_ALIGN_MAX * s

def tau_schedule_hold(epoch, total, tau_start=2.5, tau_end=0.05, hold_frac=0.3):
    hold = int(hold_frac * max(1, total - 1))
    if epoch <= hold:
        return tau_start
    s = (epoch - hold) / max(1, total - 1 - hold)
    return tau_start * (tau_end / tau_start) ** s

def build_t_weights(T, alpha=TW_ALPHA, device="cpu", dtype=torch.float32):
    t = torch.arange(T, device=device, dtype=dtype)
    w = torch.exp(-t / alpha)              # large at small t
    return w / (w.sum() + 1e-12)

def ade_fde_per_mode_weighted(pred_abs_k, gt_abs, t_w):
    # pred_abs_k: [B,K,T,2], gt_abs: [B,T,2], t_w: [T]
    d = torch.linalg.norm(pred_abs_k - gt_abs[:, None, :, :], dim=-1)  # [B,K,T]
    ade_k = (d * t_w.view(1,1,-1)).sum(dim=-1) / (t_w.sum() + 1e-12)   # [B,K]
    fde_k = torch.linalg.norm(pred_abs_k[:, :, -1, :] - gt_abs[:, -1, :][:, None, :], dim=-1)
    return ade_k, fde_k

def prefix_var_loss(pred_abs_k, last_pos, L=L_PREFIX):
    if L <= 0: return pred_abs_k.new_zeros(())
    rel = pred_abs_k - last_pos[:, None, None, :]   # [B,K,T,2]
    p = rel[:, :, :max(1, L), :]                    # [B,K,L,2]
    return ((p - p.mean(dim=1, keepdim=True))**2).sum(dim=-1).mean()

def heading_align_loss(pred_abs_k, gt_abs, last_pos, eps=1e-6):
    v_gt   = gt_abs[:, 0, :] - last_pos                          # [B,2]
    spd    = torch.norm(v_gt, dim=-1)                            # [B]
    mask   = (spd > 1e-3).float().view(-1,1)                     # avoid near-zero
    v_gt_n = F.normalize(v_gt, dim=-1, eps=eps)                  # [B,2]
    v_pred = pred_abs_k[:, :, 0, :] - last_pos[:, None, :]       # [B,K,2]
    v_pred = F.normalize(v_pred, dim=-1, eps=eps)                # [B,K,2]
    cos = (v_pred * v_gt_n[:, None, :]).sum(dim=-1).clamp(-1,1)  # [B,K]
    loss = (1.0 - cos).mean()
    # downweight examples with tiny first-step speed (pedestrians stopped, etc.)
    return (loss * mask.mean()).clamp_min(0.0)

def annealed_wta_loss_softprefix(pred_abs_k, mode_logits, gt_abs, last_pos, epoch, total_epochs):
    # ---- yaw align for the loss (recommended) ----
    if ENABLE_YAW_ALIGN:
        X_pred, X_gt = yaw_align_for_loss(pred_abs_k, gt_abs, last_pos)   # [B,K,T,2], [B,T,2]
    else:
        X_pred = pred_abs_k - last_pos[:, None, None, :]
        X_gt   = gt_abs      - last_pos[:, None, :]

    # time-weighted ADE + standard FDE in the chosen frame
    T  = X_pred.size(2)
    tw = build_t_weights(T, device=X_pred.device, dtype=X_pred.dtype)
    ade_k, fde_k = ade_fde_per_mode_weighted(X_pred, X_gt, tw)
    dist = ade_k + fde_k

    # aWTA with early tau-hold
    tau = tau_schedule_hold(epoch, total_epochs, tau_start=2.5, tau_end=0.05, hold_frac=0.3)
    w   = torch.softmax(-dist / max(tau, 1e-6), dim=1)

    # regression mixture
    reg = (w * ade_k).sum(dim=1).mean() + (w * fde_k).sum(dim=1).mean()

    # classifier CE (lighter early, detach targets early)
    s = epoch / max(1, total_epochs - 1)
    ce_weight  = 0.05 if s < 0.3 else 0.10
    ce_targets = w.detach() if s < 0.3 else w
    ce = soft_cross_entropy(mode_logits, ce_targets, detach_targets=True)

    # soft prefix regularizers in the same (aligned) frame
    w_pfx, w_head = soft_weights_for_epoch(epoch)

    def prefix_var_loss_frame(X_pred_frame, L=L_PREFIX):
        if L <= 0: return X_pred_frame.new_zeros(())
        p = X_pred_frame[:, :, :max(1, L), :]                      # [B,K,L,2]
        return ((p - p.mean(dim=1, keepdim=True))**2).sum(dim=-1).mean()

    loss = reg + ce_weight * ce + w_pfx * prefix_var_loss_frame(X_pred, L=L_PREFIX)

    # Heading align becomes redundant when yaw-aligned; only add it if not aligning
    if (not ENABLE_YAW_ALIGN) and (W_HEAD_ALIGN_MAX > 0):
        loss = loss + w_head * heading_align_loss(pred_abs_k, gt_abs, last_pos)
    
        # ---- tiny bootstrap for first-step magnitude (epochs 0–1) ----
    if epoch < 2:
        # average first step across modes in the aligned frame
        v_pred = X_pred.mean(dim=1)[:, 0, :]   # [B,2]
        v_gt   = X_gt[:, 0, :]                 # [B,2]
        loss = loss + 0.02 * F.l1_loss(v_pred, v_gt)  # small! (0.01–0.05 works)


    # logging
    best_idx = ade_k.argmin(dim=1)
    B = X_gt.size(0); row = torch.arange(B, device=X_gt.device)
    min_ade = ade_k[row, best_idx].mean()
    min_fde = fde_k[row, best_idx].mean()
    return min_ade, min_fde, loss


def run_epoch(backbone, head, loader, device,
              optimizer=None, mr_radius=2.0, ce_weight=0.1,   # ce_weight kept for API compat
              epoch: int = 0, total_epochs: int = 1,
              tau_start: float = 2.0, tau_end: float = 0.05):
    """
    Ramp ON version:
      - Applies branch ramp only if ENABLE_BRANCH_RAMP and epoch >= WARMUP_NO_RAMP_EPOCHS
      - Uses soft-prefix annealed WTA loss (time-weighted ADE, tiny prefix variance & heading alignment,
        tau-hold early, lighter CE early). Works fine together with the ramp.
    """
    train_mode = optimizer is not None
    if train_mode:
        backbone.train(); head.train()
    else:
        backbone.eval(); head.eval()

    total = 0
    sum_ade = sum_fde = sum_made = sum_mfde = sum_mr = 0.0

    for batch in loader:
        frames     = batch["frames"].to(device, non_blocking=True)
        init_masks = batch["init_masks"]
        init_labels= batch["init_labels"]
        gt_future  = batch["traj"].to(device, non_blocking=True)
        last_pos   = batch["last_pos"].to(device, non_blocking=True)

        # Forward
        feats = backbone(frames, init_masks=init_masks, init_labels=init_labels).detach()
        traj_res_k, mode_logits = head(feats)                      # [B,K,T,2]

        visualize_xmem_masks_simple(
            backbone, frames,
            save_path=f"runs/vis_masks_epoch_{epoch}.png",
            max_samples=4, max_frames=6, thresh=0.5
        )


        # --- branch ramp (hard tie) only after warm-up ---
        use_ramp_now = ENABLE_BRANCH_RAMP and (epoch >= WARMUP_NO_RAMP_EPOCHS)
        if use_ramp_now:
            L, sharp = ramp_schedule(epoch, total_epochs)
            traj_res_k = apply_branch_ramp(traj_res_k, L=L, sharp=sharp)

        pred_abs_k = last_pos[:, None, None, :] + traj_res_k       # [B,K,T,2] absolute coords

        if train_mode:
            # Soft-prefix annealed WTA (implements tau-hold, time-weighted ADE, prefix losses, CE scheduling)
            ade, fde, loss = annealed_wta_loss_softprefix(
                pred_abs_k, mode_logits, gt_future, last_pos,
                epoch=epoch, total_epochs=total_epochs
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Log world-frame best-of-K metrics
            m = {
                "ADE": ade.item(),
                "FDE": fde.item(),
                "mADE": ade.item(),
                "mFDE": fde.item(),
                "MR@2m": metrics_best_of_k(pred_abs_k, gt_future, r=mr_radius)["MR@2m"],
            }
        else:
            with torch.no_grad():
                m = metrics_best_of_k(pred_abs_k, gt_future, r=mr_radius)

        bsz = gt_future.shape[0]
        sum_ade  += m["ADE"]   * bsz
        sum_fde  += m["FDE"]   * bsz
        sum_made += m["mADE"]  * bsz
        sum_mfde += m["mFDE"]  * bsz
        sum_mr   += m["MR@2m"] * bsz
        total    += bsz

    return {
        "ADE":  sum_ade  / max(1, total),
        "FDE":  sum_fde  / max(1, total),
        "mADE": sum_made / max(1, total),
        "mFDE": sum_mfde / max(1, total),
        "MR@2m": sum_mr  / max(1, total),
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
    # for _, p in xmem_core.decoder.named_parameters():
    #     p.requires_grad = True
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
    # assert any(p.requires_grad for p in xmem_core.decoder.parameters())

    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    os.makedirs("runs", exist_ok=True)
    EPOCHS = 20
    for ep in range(EPOCHS):
        train_m = run_epoch(
            backbone, head, train_loader, device,
            optimizer=optim_all, mr_radius=2.0, ce_weight=0.1,
            epoch=ep, total_epochs=EPOCHS, tau_start=2.0, tau_end=0.05
        )
        val_m   = run_epoch(backbone, head, val_loader, device, optimizer=None, mr_radius=2.0)
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
                frames     = batch["frames"].to(device, non_blocking=True)
                init_masks = batch["init_masks"]
                init_labels= batch["init_labels"]
                gt_future  = batch["traj"].to(device, non_blocking=True)
                last_pos   = batch["last_pos"].to(device, non_blocking=True)

                feats = backbone(frames, init_masks=init_masks, init_labels=init_labels)
                traj_res_k, mode_logits = head(feats)                  # [B,K,T,2]

                        # --- branch ramp (hard tie) only after warm-up ---
                use_ramp_now = ENABLE_BRANCH_RAMP and (ep >= WARMUP_NO_RAMP_EPOCHS)
                if use_ramp_now:
                    L, sharp = ramp_schedule(ep, EPOCHS)
                    traj_res_k = apply_branch_ramp(traj_res_k, L=L, sharp=sharp)

                pred_abs_k = last_pos[:, None, None, :] + traj_res_k   # absolute coords

                visualize_predictions(
                    pred_abs_k, gt_future, last_pos,
                    save_path=f"runs/vis_epoch_{ep}.png", max_samples=8
                )
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
