# models/model_reasonnet_style.py
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

# TODO: SOLVE NORMALIZARION, deal with optimizer, update collate, mask=union detachment in predictior 

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

    def forward(self, batch):
        # inputs come from your dataloader (already using config H/cw etc.)
        cam_imgs = batch["cam_imgs"].to(self.device, non_blocking=True)             # (B,T,C,3,H,cw)
        cam_K    = batch["cam_K_scaled"].to(self.device, non_blocking=True)
        cam_T    = batch["cam_T_cam_from_ego"].to(self.device, non_blocking=True)
        cam_dep  = batch["cam_depths"].to(self.device, non_blocking=True)
        lidar    = batch["lidar_bev_raw"].to(self.device, non_blocking=True)        # (B,T,4,H_bev,W_bev)
        init_labels = batch["init_labels"]
        init0    = batch["bev_target_mask"][:, :1, 0].to(self.device).float()       # (B,1,H_bev,W_bev)

        # RGB → BEV → 3ch frames
        H_img, W_img = cam_imgs.shape[-2], cam_imgs.shape[-1]
        F_cam      = self.rgb_bev(cam_imgs, cam_dep, cam_K, cam_T, H_img, W_img)    # (B,T,C_r,H_bev,W_bev)
        frames_cam = self.cam_to_frames(F_cam)                                      # (B,T,3,H_bev,W_bev)

        # LiDAR raw → 3ch frames
        frames_lidar = self.lidar_to_frames(lidar)                                  # (B,T,3,H_bev,W_bev)

        # ReasonNet flow inside XMem
        seq_feats = self.xmem.forward(frames_cam, frames_lidar, init_masks=init0, init_labels=init_labels)  # (B,T,D)

        # head
        traj_res_k, mode_logits = self.head(seq_feats)
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
