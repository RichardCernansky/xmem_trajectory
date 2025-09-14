import os, sys, torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.network import XMem
from xmem_mm_config import xmem_mm_config
from traj.head import TrajectoryHead
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

def ade_fde_loss(pred_abs, gt_abs):
    diff = pred_abs - gt_abs
    ade = diff.norm(dim=-1).mean()
    fde = diff[:, -1].norm(dim=-1).mean()
    return ade, fde, ade + fde

@torch.no_grad()
def metrics_single(pred_abs, gt_abs, r=2.0):
    diff = pred_abs - gt_abs
    d = torch.linalg.norm(diff, dim=-1)
    ade = d.mean().item()
    fde = torch.linalg.norm(diff[:, -1], dim=-1).mean().item()
    mr = (torch.linalg.norm(diff[:, -1], dim=-1) > r).float().mean().item()
    return {"ADE": ade, "FDE": fde, "mADE": ade, "mFDE": fde, "MR@2m": mr}

def run_epoch(backbone, head, loader, device, optimizer=None, mr_radius=2.0):
    train_mode = optimizer is not None
    if train_mode:
        backbone.train(); head.train()
    else:
        backbone.eval(); head.eval()
    total = 0
    sum_ade = 0.0
    sum_fde = 0.0
    sum_made = 0.0
    sum_mfde = 0.0
    sum_mr = 0.0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        lidar_maps = batch["lidar_maps"].to(device, non_blocking=True)
        init_masks = batch["init_masks"]
        init_labels = batch["init_labels"]
        gt_future = batch["traj"].to(device, non_blocking=True)
        last_pos = batch["last_pos"].to(device, non_blocking=True)

        print("frames:", frames.device, "lidar:", lidar_maps.device)
        print("gt_future:", gt_future.device, "last_pos:", last_pos.device)
            
        if train_mode:
            feats = backbone(frames, init_masks=init_masks, init_labels=init_labels, lidar_maps=lidar_maps)

            print("feats.shape:", feats.shape)   # expect [B, T, 64]
            print("feats:", feats.device)

            pred_offsets = head(feats)
            pred_abs = last_pos.unsqueeze(1) + pred_offsets.cumsum(dim=1)
            ade, fde, loss = ade_fde_loss(pred_abs, gt_future)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            m = {"ADE": ade.item(), "FDE": fde.item(), "mADE": ade.item(), "mFDE": fde.item(), "MR@2m": metrics_single(pred_abs, gt_future, r=mr_radius)["MR@2m"]}
        else:
            with torch.no_grad():
                feats = backbone(frames, init_masks=init_masks, init_labels=init_labels, lidar_maps=lidar_maps)
                pred_offsets = head(feats)
                pred_abs = last_pos.unsqueeze(1) + pred_offsets.cumsum(dim=1)
                m = metrics_single(pred_abs, gt_future, r=mr_radius)
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=False)

    train_ds = NuScenesSeqLoader(index_path="train_agents_index.pkl", resize=True, resize_wh=(960, 540), nusc=nusc)
    val_ds = NuScenesSeqLoader(index_path="val_agents_index.pkl", resize=True, resize_wh=(960, 540), nusc=nusc)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_varK)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_varK)

    xmem_core = load_xmem(device=device)
    for p in xmem_core.parameters():
        p.requires_grad = False
    xmem_core.eval()

    mm_cfg = xmem_mm_config(hidden_dim=getattr(xmem_core, "hidden_dim", 256))
    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device, n_lidar=5, fusion_mode="concat", use_bn=False).to(device)
    head = TrajectoryHead(d_in=getattr(xmem_core, "hidden_dim", 256), d_hid=256, horizon=30).to(device)

    optim_all = optim.AdamW([p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad], lr=1e-3)

    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    epochs = 20
    for ep in range(epochs):
        train_m = run_epoch(backbone, head, train_loader, device, optimizer=optim_all, mr_radius=2.0)
        val_m = run_epoch(backbone, head, val_loader, device, optimizer=None, mr_radius=2.0)
        print(f"Epoch {ep}: Train ADE {train_m['ADE']:.3f} FDE {train_m['FDE']:.3f} | Val ADE {val_m['ADE']:.3f} FDE {val_m['FDE']:.3f} MR@2m {val_m['MR@2m']:.3f}")
        hist["epoch"].append(ep)
        hist["train_ADE"].append(train_m["ADE"])
        hist["train_FDE"].append(train_m["FDE"])
        hist["val_ADE"].append(val_m["ADE"])
        hist["val_FDE"].append(val_m["FDE"])
        hist["val_MR2"].append(val_m["MR@2m"])

    os.makedirs("runs", exist_ok=True)
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
