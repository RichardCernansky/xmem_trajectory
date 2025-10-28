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

    def forward(self, batch):
        dev = self.device

        # --- Get all fields from batch (keep on CPU for now) ---
        cam_imgs = batch["cam_imgs"]            # (B,T,Cams,3,H,W)
        cam_K    = batch["cam_K_scaled"]
        cam_T    = batch["cam_T_cam_from_ego"]
        cam_dep  = batch["cam_depths"]
        lidar    = batch["lidar_bev_raw"]       # (B,T,4,H_bev,W_bev)
        init_labels = batch["init_labels"]      # stays on CPU or small list
        init0    = batch["bev_target_mask"][:, :1, 0].float()   # (B,1,H_bev,W_bev)

        # --- Static image size ---
        H_img, W_img = cam_imgs.shape[-2:]
        B, T = cam_imgs.shape[:2]

        seq_feats = []
        # --- Process each timestep sequentially ---
        print("hello another forward")
        for t in range(T):
            # Move only current timestep to GPU
            cam_imgs_t = cam_imgs[:, t].to(dev, non_blocking=True)
            cam_K_t    = cam_K[:, t].to(dev, non_blocking=True)
            cam_T_t    = cam_T[:, t].to(dev, non_blocking=True)
            cam_dep_t  = cam_dep[:, t].to(dev, non_blocking=True)
            lidar_t    = lidar[:, t].to(dev, non_blocking=True)

           # --- RGB → BEV ---
            F_cam_t = self.rgb_bev(
                cam_imgs_t.unsqueeze(1),   # add fake time dimension for encoder
                cam_dep_t.unsqueeze(1),
                cam_K_t.unsqueeze(1),
                cam_T_t.unsqueeze(1),
                H_img, W_img
            )                              # (B, 1, C_r, Hb, Wb)

            # convert BEV features → 3-channel frame, then remove T dimension
            frames_cam_t = senorlf.cam_to_frames(F_cam_t)[:, 0]   # (B, 3, Hb, Wb)

            # --- LiDAR → 3-channel BEV ---
            frames_lidar_t = self.lidar_to_frames(lidar_t.unsqueeze(1))[:, 0]  # (B, 3, Hb, Wb)

            # --- Feed both modalities into XMem for this timestep ---
            feats_t = self.xmem.forward_step(t, 
                frames_cam_t, frames_lidar_t,
                init_masks=init0.to(dev, non_blocking=True),
                init_labels=init_labels
            )
            seq_feats.append(feats_t)

            # free GPU memory before next frame
            # del cam_imgs_t, cam_K_t, cam_T_t, cam_dep_t, lidar_t, F_cam_t, frames_cam_t, frames_lidar_t
            # torch.cuda.empty_cache()

        # --- Stack timestep features ---
        seq_feats = torch.stack(seq_feats, dim=1)  # (B,T,D)

        # --- Predict trajectories ---
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
