
import torch, torch.nn as nn
import os
from trainer.utils import open_config
from data.configs.filenames import PP_CONFIG, PP_CHECKPOINT

from my_model.lidar.pp_loader import build_pointpillars, load_pp_backbone_weights
from my_model.adapters.pp_to_frames import PPToFrames 

from .xmem_wrapper.predictor import XMemBackboneWrapper
from .head import MultiModalTrajectoryHead
from .optimizer import make_optimizer
from .losses import best_of_k_loss, mask_loss
from .metrics import metrics_best_of_k
from visualizer.visualizer import visualize



class MemoryModel(nn.Module):
    def __init__(self, device: str, vis_path, train_config: dict):
        super().__init__()
        self.device = device
        self.train_config = train_config
        self.pp_config = open_config(PP_CONFIG)
        self.vis_path = vis_path
        
        self.first_stage = self.train_config["first_stage"]
        self.K         = int(self.train_config["K"])
        self.horizon   = int(self.train_config["horizon"])
        self.mr_radius = float(self.train_config["mr_radius"])

        self.pp = build_pointpillars(self.pp_config, device)
        _ = load_pp_backbone_weights(self.pp, PP_CHECKPOINT)
        self.pp.to(device)
        self.pp.eval()
        for p in self.pp.parameters():
            p.requires_grad = False

        self.pp_to_frames = PPToFrames(c_in= self.pp.pts_neck.out_channels if self.pp.pts_neck else 256).to(device)

        # XMem (uses its own flags from your wrapper; frozen or trainable via config)
        self.xmem = XMemBackboneWrapper(device, self.train_config)
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
        

    # in forward
    def forward(self, batch):
        dev = self.device

        bev_masks_all = batch["bev_target_mask"].to(dev, non_blocking=True).float()
        init_labels = torch.as_tensor(batch["init_labels"]).to(
            dev, non_blocking=True
        ).long()
        lidar = batch["points"].to(dev, non_blocking=True)

        B, T = lidar.shape[:2]
        seq_feats = []
        occ_logits_seq = []

        for t in range(T):
            masks = bev_masks_all[:, t]
            points_t = lidar[:, t]
            bev_feat_t = self.pp(points_t)
            frames_lidar_t = self.pp_to_frames(bev_feat_t)

            feats_t, occ_logits_t = self.xmem.forward_step(
                t,
                frames_lidar_t,
                init_masks=masks,
                init_labels=init_labels,
            )

            seq_feats.append(feats_t)
            occ_logits_seq.append(occ_logits_t)

        occ_logits = torch.stack(occ_logits_seq, dim=1)

        traj_res_k = None
        mode_logits = None

        if self.head is not None:
            seq_feats = torch.stack(seq_feats, dim=1)
            traj_res_k, mode_logits = self.head(seq_feats)

        return traj_res_k, mode_logits, occ_logits



    def to_abs(self, traj_res_k, last_pos):
        return last_pos[:, None, None, :] + traj_res_k

    def training_step(self, batch, epoch: int):
        self.train()

        bev_masks_all = batch["bev_target_mask"].to(self.device, non_blocking=True).float()

        traj_res_k, mode_logits, occ_logits = self.forward(batch)
        visualize(epoch, batch, self.pp, occ_logits.detach(), bev_masks_all.detach())

        loss_mask = mask_loss(occ_logits, bev_masks_all)
        metrics = {"mask_loss": loss_mask.item()}

        if self.first_stage:
            loss = loss_mask
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            return metrics, None

        gt_future = batch["traj"].to(self.device, non_blocking=True)
        last_pos = batch["last_pos"].to(self.device, non_blocking=True)

        pred_abs_k = self.to_abs(traj_res_k, last_pos)
        ade, fde, loss = best_of_k_loss(pred_abs_k, mode_logits, gt_future)


        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        mr = metrics_best_of_k(pred_abs_k, gt_future, r=self.mr_radius)["MR@2m"]

        metrics.update(
            {
                "ADE": ade.item(),
                "FDE": fde.item(),
                "mADE": ade.item(),
                "mFDE": fde.item(),
                "MR@2m": mr,
                "loss": loss.item(),
            }
        )

        return metrics, pred_abs_k.detach()

    def validation_step(self, batch):
        self.eval()
        with torch.inference_mode():
            bev_masks_all = batch["bev_target_mask"].to(
                self.device, non_blocking=True
            ).float()
            traj_res_k, mode_logits, occ_logits = self.forward(batch)

            loss_mask = mask_loss(occ_logits, bev_masks_all)
            metrics = {"mask_loss": loss_mask.item()}

            if self.first_stage:
                return metrics, None

            gt_future = batch["traj"].to(self.device, non_blocking=True)
            last_pos = batch["last_pos"].to(self.device, non_blocking=True)

            pred_abs_k = self.to_abs(traj_res_k, last_pos)
            m = metrics_best_of_k(pred_abs_k, gt_future, r=self.mr_radius)
            m["mask_loss"] = loss_mask.item()
            return m, pred_abs_k
