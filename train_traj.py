import os, sys, torch
import torch.optim as optim

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
    """
    Proper loader that reads dims from the checkpoint.
    """
    cfg = {"single_object": False}
    net = XMem(cfg, model_path=backbone_ckpt, map_location="cpu")
    state = torch.load(backbone_ckpt, map_location="cpu")  # weights_only warning is fine
    net.load_weights(state, init_as_zero_if_needed=True)
    net.to(device).eval()
    return net

def ade_fde_loss(pred_abs, gt_abs):
    diff = pred_abs - gt_abs
    ade = diff.norm(dim=-1).mean()
    fde = diff[:, -1].norm(dim=-1).mean()
    return ade, fde, ade + fde


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=False)
    train_ds = NuScenesSeqLoader(index_path="train_agents_index.pkl", resize=True, resize_wh=(960, 540), nusc=nusc)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_varK)

    xmem_core = load_xmem(device=device)
    mm_cfg = xmem_mm_config(xmem_core)

    backbone = XMemMMBackboneWrapper(mm_cfg=mm_cfg, xmem=xmem_core, device=device, n_lidar=5, fusion_mode="concat", use_bn=False)
    head = TrajectoryHead(d_in=getattr(xmem_core, "hidden_dim", 256), d_hid=256, horizon=30).to(device)

    optim_all = optim.AdamW([p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad], lr=1e-3)

    for epoch in range(5):
        for batch in train_loader:
            frames      = batch["frames"].to(device)
            lidar_maps  = batch["lidar_maps"].to(device)
            init_masks  = batch["init_masks"]
            init_labels = batch["init_labels"]
            gt_future   = batch["traj"].to(device)
            last_pos    = batch["last_pos"].to(device)

            with torch.no_grad():
                feats = backbone(frames, init_masks=init_masks, init_labels=init_labels, lidar_maps=lidar_maps)

            pred_offsets = head(feats)
            pred_abs = last_pos.unsqueeze(1) + pred_offsets.cumsum(dim=1)

            ade, fde, loss = ade_fde_loss(pred_abs, gt_future)
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()

        print(f"Epoch {epoch}: ADE {ade.item():.3f} | FDE {fde.item():.3f}")

if __name__ == "__main__":
    main()
