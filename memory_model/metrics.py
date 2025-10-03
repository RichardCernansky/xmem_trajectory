import torch

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
