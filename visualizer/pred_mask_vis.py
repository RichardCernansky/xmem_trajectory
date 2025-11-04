# visualization/xmem_writes.py
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _minmax01(x: torch.Tensor) -> torch.Tensor:
    m = x.amin(dim=(-2,-1), keepdim=True)
    M = x.amax(dim=(-2,-1), keepdim=True)
    return (x - m) / (M - m + 1e-8)

@torch.no_grad()
def save_write_events(write_events, lidar_frames_all, bev_masks_all, outdir="viz/forward_writes", limit=60, dpi=140):
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    made = 0
    B, T, _, Hb, Wb = lidar_frames_all.shape
    for ev in write_events:
        if ev is None:
            continue
        if made >= limit:
            break
        b = int(ev["b"]); t = int(ev["t"])
        if not (0 <= b < B and 0 <= t < T):
            continue
        tw = ev["to_write"]
        if isinstance(tw, torch.Tensor) and tw.is_cuda:
            tw = tw.cpu()
        if tw.dim() != 3 or tw.size(0) <= 1:
            continue
        fg = tw[1:].sum(0, keepdim=True).clamp(0, 1)
        fg = F.interpolate(fg.unsqueeze(0), size=(Hb, Wb), mode="bilinear", align_corners=False)[0,0]
        gt = bev_masks_all[b, t, 0].float()
        base = _minmax01(lidar_frames_all[b, t]).mean(0)
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax in axs: ax.axis("off")
        axs[0].imshow(fg.numpy(), cmap="magma", interpolation="nearest"); axs[0].set_title(f"to_write FG  t={t} b={b} ({ev.get('source','?')})")
        axs[1].imshow(gt.numpy(), cmap="gray", interpolation="nearest");   axs[1].set_title("GT mask")
        axs[2].imshow(base.numpy(), cmap="gray", interpolation="nearest")
        axs[2].imshow(fg.numpy(), cmap="jet", alpha=0.45, interpolation="nearest"); axs[2].set_title("Overlay")
        fname = out / f"write_t{t:03d}_b{b:02d}_{made:05d}.png"
        plt.tight_layout(); fig.savefig(fname, dpi=dpi); plt.close(fig)
        made += 1
