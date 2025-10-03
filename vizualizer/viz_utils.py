import torch

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