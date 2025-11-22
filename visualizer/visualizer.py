import torch, torch.nn as nn
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import torch.nn.functional as F


def visualize_masks(epoch: int, occ_logits: torch.Tensor, bev_masks_all: torch.Tensor) -> None:
    b = 0
    pred = torch.sigmoid(occ_logits[b, :, 0])
    gt = bev_masks_all[b, :, 0].float()

    if pred.shape[-2:] != gt.shape[-2:]:
        H, W = pred.shape[-2:]
        gt = F.interpolate(gt.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)

    T = pred.size(0)
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6), dpi=300)
    if T == 1:
        axes = axes.reshape(2, 1)

    for t in range(T):
        p_img = pred[t].detach().cpu().numpy()
        g_img = gt[t].detach().cpu().numpy()

        ax_p = axes[0, t]
        ax_g = axes[1, t]

        ax_p.imshow(p_img, cmap="viridis")
        ax_p.set_title(f"pred t={t}")
        ax_p.axis("off")

        ax_g.imshow(g_img, cmap="gray")
        ax_g.set_title(f"gt t={t}")
        ax_g.axis("off")

    os.makedirs("out/xmem_pred", exist_ok=True)
    out_path = os.path.join("out/xmem_pred", f"pred_vs_gt_epoch{epoch}.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def visualize(
    epoch: int,
    batch: dict,
    pp_model: nn.Module,
    occ_logits: torch.Tensor,
    bev_masks_all: torch.Tensor,
) -> None:
    b = 0
    points = batch["points"][b]
    T = points.shape[0]

    device = next(pp_model.parameters()).device
    bev_list = []
    for t in range(T):
        pts_t = points[t].unsqueeze(0).to(device)
        bev_t = pp_model(pts_t)
        bev_list.append(bev_t[0].detach().cpu())
    bev_feats = torch.stack(bev_list, dim=0)

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6), dpi=300)
    if T == 1:
        axes = axes.reshape(2, 1)

    for t in range(T):
        pts = points[t]
        x = pts[:, 0].detach().cpu().numpy()
        y = pts[:, 1].detach().cpu().numpy()

        ax_l = axes[0, t]
        ax_pp = axes[1, t]

        ax_l.scatter(x, y, s=0.3)
        ax_l.set_title(f"lidar t={t}")
        ax_l.axis("equal")

        bev_img = bev_feats[t].mean(0).numpy()
        ax_pp.imshow(bev_img, cmap="viridis")
        ax_pp.set_title(f"pp bev t={t}")
        ax_pp.axis("off")

    os.makedirs("out/lidar_pp", exist_ok=True)
    out_path = os.path.join("out/lidar_pp", f"lidar_pp_epoch{epoch}.pdf")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    visualize_masks(epoch, occ_logits, bev_masks_all)
