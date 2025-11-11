import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_steps(
    write_events,
    lidar_frames_all,   # Tensor [B,T,3,H,W]  (you already have this)
    bev_masks_all,      # Tensor [B,T,1,H,W]
    outpath,            # e.g. "data/vis/window_t5_9.png"
    b=0,
    start=5,
    stop=9,
    dpi=140,
    backbone_last_masks=None  # optional: [B,T,H,W] or [T,H,W] fallback if no write event
):
    """
    One figure with columns for t=start..stop (default 5..9).
    Row0: predicted 'to_write' FG (if write happened at t; otherwise zeros or fallback mask)
    Row1: GT mask
    Row2: Overlay (LiDAR gray + FG heatmap)
    """
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    # Move to CPU numpy
    lidar_frames_all = lidar_frames_all.detach().cpu()
    bev_masks_all    = bev_masks_all.detach().cpu()

    B, T, C, H, W = lidar_frames_all.shape
    start = max(0, start)
    stop  = min(stop, T - 1)
    cols  = stop - start + 1

    # Build an index for quick event lookup
    ev_map = {}
    for e in write_events:
        if e is None: 
            continue
        tb = (int(e["t"]), int(e["b"]))
        ev_map[tb] = e  # last-one-wins if duplicates

    # Normalize BEV background (use channel 0)
    def to01(x):
        x = x.astype(np.float32)
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    # Optional fallback masks from backbone (if provided)
    fallback = None
    if backbone_last_masks is not None:
        fb = backbone_last_masks
        if isinstance(fb, torch.Tensor):
            fb = fb.detach().cpu().numpy()
        if fb.ndim == 4:       # [B,T,H,W]
            fallback = fb[b]
        elif fb.ndim == 3:     # [T,H,W]
            fallback = fb
        else:
            fallback = None

    fig, axes = plt.subplots(3, cols, figsize=(3.8*cols, 10))
    if cols == 1:
        axes = np.array([[axes[0]],[axes[1]],[axes[2]]])  # ensure 2D indexing

    for j, t in enumerate(range(start, stop + 1)):
        ax_pred = axes[0, j]
        ax_gt   = axes[1, j]
        ax_ov   = axes[2, j]

        base = lidar_frames_all[b, t, 0].numpy()
        base = to01(base)

        # Default FG = zeros; replace if we have a write at this (b,t)
        fg = np.zeros((H, W), dtype=np.float32)
        src = "none"
        e = ev_map.get((t, b))
        if e is not None and isinstance(e.get("to_write"), torch.Tensor):
            tw = e["to_write"]  # (K+1,H,W) on CPU from your code
            if tw.ndim == 3 and tw.shape[0] > 1:
                fg = tw[1].numpy().astype(np.float32)  # foreground channel
                src = e.get("source", "pred/gt")
        elif fallback is not None:
            # show the model’s last predicted mask if available
            fg = fallback[t].astype(np.float32)
            src = "fallback"

        gt = bev_masks_all[b, t, 0].numpy().astype(np.float32)

        # Row 1: predicted write FG
        im0 = ax_pred.imshow(fg, cmap="magma")
        ax_pred.set_title(f"t={t}  pred ({src})")
        ax_pred.axis("off")

        # Row 2: GT mask
        ax_gt.imshow(gt, cmap="gray")
        ax_gt.set_title("GT mask")
        ax_gt.axis("off")

        # Row 3: overlay
        ax_ov.imshow(base, cmap="Greys")
        ax_ov.imshow(fg, cmap="magma", alpha=0.65)
        ax_ov.set_title("Overlay")
        ax_ov.axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
