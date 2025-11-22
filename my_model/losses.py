import torch
from my_model.metrics import ade_fde_per_mode
import torch.nn.functional as F

def best_of_k_loss(pred_abs_k, logits, gt_abs, ce_weight=0.1, fde_weight=2.0):
    ade_k, fde_k = ade_fde_per_mode(pred_abs_k, gt_abs)
    best_idx = ade_k.argmin(dim=1)
    B = gt_abs.size(0)
    row = torch.arange(B, device=gt_abs.device)
    min_ade = ade_k[row, best_idx].mean()
    min_fde = fde_k[row, best_idx].mean()
    ce = torch.nn.functional.cross_entropy(logits, best_idx)
    total_loss = min_ade + fde_weight * min_fde + ce_weight * ce
    return min_ade, min_fde, total_loss

def mask_loss(occ_logits, target_masks, mode: str = "bce_dice", smooth: float = 1e-5):
    occ_logits = occ_logits.float()
    target_masks = target_masks.float()

    B, T, C, H_pred, W_pred = occ_logits.shape
    _, T_gt, C_gt, H_gt, W_gt = target_masks.shape

    if (H_gt, W_gt) != (H_pred, W_pred):
        target_masks = F.interpolate(
            target_masks.view(B * T, C, H_gt, W_gt),
            size=(H_pred, W_pred),
            mode="nearest",
        ).view(B, T, C, H_pred, W_pred)

    if mode == "bce":
        return F.binary_cross_entropy_with_logits(occ_logits, target_masks)

    prob = torch.sigmoid(occ_logits)

    if mode == "dice":
        inter = (prob * target_masks).sum(dim=(2, 3, 4))
        union = prob.sum(dim=(2, 3, 4)) + target_masks.sum(dim=(2, 3, 4)) + smooth
        dice = 1.0 - (2.0 * inter + smooth) / union
        return dice.mean()

    if mode == "bce_dice":
        bce = F.binary_cross_entropy_with_logits(occ_logits, target_masks)
        inter = (prob * target_masks).sum(dim=(2, 3, 4))
        union = prob.sum(dim=(2, 3, 4)) + target_masks.sum(dim=(2, 3, 4)) + smooth
        dice = 1.0 - (2.0 * inter + smooth) / union
        return bce + dice.mean()

    raise ValueError(f"Unknown mask loss mode: {mode}")
