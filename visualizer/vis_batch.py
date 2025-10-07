#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nuscenes.nuscenes import NuScenes
from datamodule.datamodule import NuScenesDataModule
from trainer.utils import open_config, open_index
from data.configs.filenames import TRAIN_CONFIG, TRAIN_INDEX, VAL_INDEX

# ---------- helpers ----------
def _to_img01(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.clamp(0, 1).permute(1, 2, 0).numpy()

def _depth_to_rgb(depth: np.ndarray, dmax: float, cmap_name: str) -> np.ndarray:
    dnorm = np.clip(depth / max(dmax, 1e-6), 0.0, 1.0)
    rgb = cm.get_cmap(cmap_name)(dnorm)[..., :3]
    rgb[depth <= 0] = 0.0
    return rgb

def overlay_heatmap(frame_3chw: torch.Tensor, depth_1hw: torch.Tensor,
                    dmax: float = 80.0, alpha: float = 0.55, cmap: str = 'viridis') -> np.ndarray:
    base = _to_img01(frame_3chw)                                # (H,W,3)
    depth = depth_1hw.detach().cpu().float().squeeze(0).numpy() # (H,W)
    mask = depth > 0
    depth_rgb = _depth_to_rgb(depth, dmax, cmap)                 # (H,W,3)
    out = base.copy()
    out[mask] = (1 - alpha) * base[mask] + alpha * depth_rgb[mask]
    return out

def overlay_points(frame_3chw: torch.Tensor, depth_1hw: torch.Tensor,
                   dmax: float = 80.0, step: int = 3, size: float = 1.0, cmap: str = 'viridis'):
    base = _to_img01(frame_3chw)
    depth = depth_1hw.detach().cpu().float().squeeze(0).numpy()
    H, W = depth.shape
    yy, xx = np.nonzero(depth > 0)
    if len(xx) == 0:
        return base
    if step > 1:
        sel = (np.arange(len(xx)) % step) == 0
        xx, yy = xx[sel], yy[sel]
    d = np.clip(depth[yy, xx] / max(dmax, 1e-6), 0.0, 1.0)

    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes([0,0,1,1])
    ax.imshow(base)
    sc = ax.scatter(xx, yy, c=d, s=size, cmap=cmap, marker='.', linewidths=0)
    ax.set_axis_off()
    fig.canvas.draw()
    # render to array (raster) if you want, but we’ll save directly from fig for true vector points
    return fig  # caller will save & close

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Save each timestep of one batch as a separate vector PDF overlay.")
    ap.add_argument("--split", type=str, default="train", choices=["train","val"])
    ap.add_argument("--version", type=str, default="v1.0-trainval")
    ap.add_argument("--dataroot", type=str, default=r"e:\nuscenes")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--index", type=int, default=0, help="sample index within the batch (0..B-1)")
    ap.add_argument("--mode", type=str, default="heatmap", choices=["heatmap","points"])
    ap.add_argument("--dmax", type=float, default=80.0)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--cmap", type=str, default="viridis")
    ap.add_argument("--outdir", type=str, default="./visualizer/sample_viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # build nuScenes + rows same as your trainer
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    _ = open_config(TRAIN_CONFIG)  # not used directly here; kept for parity
    train_rows = open_index(TRAIN_INDEX)
    val_rows   = open_index(VAL_INDEX)

    dm = NuScenesDataModule(nusc, train_rows, val_rows)
    loader = dm.train_dataloader() if args.split == "train" else dm.val_dataloader()

    # grab ONE batch
    batch = next(iter(loader))
    frames = batch["frames"][args.index]                  # (T,3,H,W)
    depths = batch["depths"][args.index]

    T = frames.shape[0]

    for t in range(T):
        if depths is not None and args.mode == "heatmap":
            over = overlay_heatmap(frames[t], depths[t], dmax=args.dmax, alpha=args.alpha, cmap=args.cmap)
            H, W, _ = over.shape
            fig = plt.figure(figsize=(W/100, H/100), dpi=100)
            ax = plt.axes([0,0,1,1])
            ax.imshow(over)
            ax.set_axis_off()
            pdf_path = os.path.join(args.outdir, f"{args.split}_b{args.index}_t{t:02d}_heatmap.pdf")
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"saved {pdf_path}")

        elif depths is not None and args.mode == "points":
            fig = overlay_points(frames[t], depths[t], dmax=args.dmax, step=3, size=1.0, cmap=args.cmap)
            pdf_path = os.path.join(args.outdir, f"{args.split}_b{args.index}_t{t:02d}_points.pdf")
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"saved {pdf_path}")

        else:
            # no depths available: save plain RGB pano
            img = _to_img01(frames[t])
            H, W, _ = img.shape
            fig = plt.figure(figsize=(W/100, H/100), dpi=100)
            ax = plt.axes([0,0,1,1])
            ax.imshow(img)
            ax.set_axis_off()
            pdf_path = os.path.join(args.outdir, f"{args.split}_b{args.index}_t{t:02d}_rgb.pdf")
            plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print(f"saved {pdf_path}")

if __name__ == "__main__":
    main()
