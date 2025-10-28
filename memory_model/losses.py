import torch
from memory_model.metrics import ade_fde_per_mode

def best_of_k_loss(pred_abs_k, probs, gt_abs, ce_weight=0.1, fde_weight=2.0):
    # pred_abs_k: [B,K,T,2], probs: [B,K], gt_abs: [B,T,2]
    ade_k, fde_k = ade_fde_per_mode(pred_abs_k, gt_abs)    # [B,K], [B,K]
    best_idx = ade_k.argmin(dim=1)                         # [B]
    B = gt_abs.size(0)
    row = torch.arange(B, device=gt_abs.device)
    
    min_ade = ade_k[row, best_idx].mean()
    min_fde = fde_k[row, best_idx].mean()
    ce = torch.nn.functional.cross_entropy(probs, best_idx)

    # combine with weights
    total_loss = min_ade + fde_weight * min_fde + ce_weight * ce
    return min_ade, min_fde, total_loss
