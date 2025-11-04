
import torch, torch.nn as nn
from trainer.utils import open_config
from data.configs.filenames import TRAIN_CONFIG
from .adapters.rgb_liftsplat import RGBLiftSplatEncoder
from .adapters.bev_into_frames import CamBEVToFrames, LiDARToFrames
from .xmem_wrapper.predictor import XMemBackboneWrapper
from .head import MultiModalTrajectoryHead
from .optimizer import make_optimizer
from .losses import best_of_k_loss
from .metrics import metrics_best_of_k
from visualizer.pred_mask_vis import save_write_events




# TODO: SOLVE NORMALIZARION, deal with optimizer, mask=union detachment in predictior , deal with deep update not working

class MemoryModel(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.train_config = open_config(TRAIN_CONFIG)

        # pull all BEV specs & training knobs from config
        Hb   = int(self.train_config["H_bev"])
        Wb   = int(self.train_config["W_bev"])
        xmn, xmx = [float(x) for x in self.train_config["bev_x_bounds"]]
        ymn, ymx = [float(y) for y in self.train_config["bev_y_bounds"]]

        self.K         = int(self.train_config["K"])
        self.horizon   = int(self.train_config["horizon"])
        self.mr_radius = float(self.train_config["mr_radius"])

        # camera branch (RGB → BEV) sized by config
        C_r = self.train_config["C_r"]
        self.rgb_bev         = RGBLiftSplatEncoder(Hb, Wb, xmn, xmx, ymn, ymx, C_r=C_r).to(device)
        self.cam_to_frames   = CamBEVToFrames(c_in=C_r).to(device)

        self.lidar_to_frames = LiDARToFrames(Hb, Wb, with_posenc=True).to(device)

        # XMem (uses its own flags from your wrapper; frozen or trainable via config)
        self.xmem = XMemBackboneWrapper(device)
        self.hidden_dim = self.xmem.hidden_dim  # D from your wrapper (e.g., 64)

        # trajectory head
        self.head = MultiModalTrajectoryHead(d_in=self.hidden_dim,
                                             t_out=self.horizon,
                                             K=self.K,
                                             hidden=self.hidden_dim).to(device)

        # optimizer uses your per-group LRs from config
        self.optimizer = make_optimizer(self)

        # --- Optimizer coverage ---
        total = 0
        for i, g in enumerate(self.optimizer.param_groups):
            n = sum(p.numel() for p in g["params"] if p.requires_grad)
            print(f"[OPT] group {i}: params={n} lr={g.get('lr')}")
            total += n
        print("[OPT] TOTAL trainable params in optimizer:", total)

        print("[OPT] head trainable:",
            sum(p.numel() for p in self.head.parameters() if p.requires_grad))
        print("[OPT] xmem.key_enc trainable:",
            sum(p.numel() for p in self.xmem.xmem_core.key_encoder.parameters()
                if p.requires_grad))
        print("[OPT] xmem.val_enc trainable:",
            sum(p.numel() for p in self.xmem.xmem_core.value_encoder.parameters()
                if p.requires_grad))
        print("[OPT] xmem.decoder trainable:",
            sum(p.numel() for p in self.xmem.xmem_core.decoder.parameters()
                if p.requires_grad))
        
        
    def forward(self, batch):
        dev = self.device

        cam_imgs = batch["cam_imgs"]            # (B,T,C,3,H,W)
        cam_K    = batch["cam_K_scaled"]
        cam_T    = batch["cam_T_cam_from_ego"]
        cam_dep  = batch["cam_depths"]
        lidar    = batch["lidar_bev_raw"]       # (B,T,4,Hb,Wb)
        init_labels = batch["init_labels"]

        H_img, W_img = cam_imgs.shape[-2:]
        B, T = cam_imgs.shape[:2]

        seq_feats = []
        write_events = []

        # Precompute LiDAR "frames" for visualization once (cheap, 1×1×1 conv)
        lidar_frames_all = self.lidar_to_frames(lidar.to(dev, non_blocking=True))  # (B,T,3,Hb,Wb)
        bev_masks_all = batch["bev_target_mask"].to(dev, non_blocking=True)        # (B,T,1,Hb,Wb)

        for t in range(T):
            cam_imgs_t = cam_imgs[:, t].to(dev, non_blocking=True)
            cam_K_t    = cam_K[:, t].to(dev, non_blocking=True)
            cam_T_t    = cam_T[:, t].to(dev, non_blocking=True)
            cam_dep_t  = cam_dep[:, t].to(dev, non_blocking=True)
            lidar_t    = lidar[:, t].to(dev, non_blocking=True)

            F_cam_t = self.rgb_bev(
                cam_imgs_t.unsqueeze(1),
                cam_dep_t.unsqueeze(1),
                cam_K_t.unsqueeze(1),
                cam_T_t.unsqueeze(1),
                H_img, W_img
            )
            frames_cam_t   = self.cam_to_frames(F_cam_t)[:, 0]              # (B,3,Hb,Wb)
            frames_lidar_t = lidar_frames_all[:, t]                         # reuse precomputed (B,3,Hb,Wb)
            init_t         = bev_masks_all[:, t].float()                    # (B,1,Hb,Wb)

            feats_t, writes_t = self.xmem.forward_step(
                t, frames_cam_t, frames_lidar_t,
                init_masks=init_t, init_labels=init_labels
            )
            seq_feats.append(feats_t)
            for w in writes_t:
                if w is not None:
                    write_events.append(w)

        seq_feats = torch.stack(seq_feats, dim=1)  # (B,T,D)
        traj_res_k, mode_logits = self.head(seq_feats)

        # ---- VISUALIZE all writes for this batch (right here) ----
        # This is unconditional (as you asked), with a cap inside to avoid spam.
        save_write_events(
            write_events,
            lidar_frames_all.detach().cpu(),
            bev_masks_all.detach().cpu(),
            outdir="data/vis/forward_write_masks",
            limit=60,
            dpi=140
        )

        return traj_res_k, mode_logits



    def to_abs(self, traj_res_k, last_pos):
        return last_pos[:, None, None, :] + traj_res_k

    def training_step(self, batch, epoch: int):
        self.train()
        gt_future = batch["traj"].to(self.device, non_blocking=True)
        last_pos  = batch["last_pos"].to(self.device, non_blocking=True)

        traj_res_k, mode_logits = self.forward(batch)
        pred_abs_k = self.to_abs(traj_res_k, last_pos)
        ade, fde, loss = best_of_k_loss(pred_abs_k, mode_logits, gt_future)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        def mean_grad(m):
            g = [p.grad.abs().mean().item()
                for p in m.parameters() if p.grad is not None]
            return sum(g)/len(g) if g else 0.0

        # print("[GRAD] head:", mean_grad(self.head))
        # print("[GRAD] xmem.key_enc:",
        #     mean_grad(self.xmem.xmem_core.key_encoder))
        # print("[GRAD] xmem.val_enc:",
        #     mean_grad(self.xmem.xmem_core.value_encoder))
        # print("[GRAD] xmem.decoder:",
        #     mean_grad(self.xmem.xmem_core.decoder))

        self.optimizer.step()

        mr = metrics_best_of_k(pred_abs_k, gt_future, r=self.mr_radius)["MR@2m"]
        return {"ADE": ade.item(), "FDE": fde.item(), "mADE": ade.item(), "mFDE": fde.item(), "MR@2m": mr, "loss": loss.item()}, pred_abs_k.detach()

    def validation_step(self, batch):
        self.eval()
        with torch.inference_mode():
            gt_future = batch["traj"].to(self.device, non_blocking=True)
            last_pos  = batch["last_pos"].to(self.device, non_blocking=True)

            traj_res_k, mode_logits = self.forward(batch)
            pred_abs_k = self.to_abs(traj_res_k, last_pos)
            m = metrics_best_of_k(pred_abs_k, gt_future, r=self.mr_radius)
            return m, pred_abs_k
