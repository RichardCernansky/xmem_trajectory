import sys
import torch
from torch import nn

from data.configs.filenames import TRAIN_CONFIG
from trainer.utils import open_config
from .optimizer import make_optimizer
from .xmem_wrapper.predictor import XMemBackboneWrapper
from .head import MultiModalTrajectoryHead
from .early_fusion import EarlyFusionAdapter
from .losses import best_of_k_loss
from .metrics import metrics_best_of_k



class MemoryModel(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.train_config = open_config(TRAIN_CONFIG)
        self.K = self.train_config["K"]
        self.horizon = self.train_config["horizon"]
        self.epochs = self.train_config["epochs"]
        self.mr_radius = self.train_config["mr_radius"]
        self.device = device

        # architecture
        self.early_fusion = EarlyFusionAdapter().to(self.device)
        self.xmem_backbone_wrapper = XMemBackboneWrapper(device)
        self.hidden_dim = self.xmem_backbone_wrapper.hidden_dim
        self.head = MultiModalTrajectoryHead(d_in=self.hidden_dim,t_out=self.horizon,K=self.K,hidden=self.hidden_dim).to(self.device)

        #create optimizer
        self.optimizer = make_optimizer(self)


    def forward(self, frames, depth_extras, *, init_masks=None, init_labels=None):
        # early fusion
        x5 = torch.cat([frames, depth_extras], dim=2) 
        frames = self.early_fusion(x5) 

        # xmem
        feats = self.xmem_backbone_wrapper(frames, init_masks=init_masks, init_labels=init_labels)

        # head
        traj_res_k, mode_logits = self.head(feats)

        return traj_res_k, mode_logits

    def predict(self, frames, init_masks=None, init_labels=None):
        with torch.inference_mode():
            self.eval()
            return self.forward(frames, init_masks=init_masks, init_labels=init_labels, detach_feats=True)

    def to_abs(self, traj_res_k: torch.Tensor, last_pos: torch.Tensor) -> torch.Tensor:
        return last_pos[:, None, None, :] + traj_res_k

    def training_step(self, batch,  epoch: int):
        self.train()
        frames      = batch["frames"].to(self.device, non_blocking=True)
        init_masks  = batch["init_masks"]
        init_labels = batch["init_labels"]
        depth_extras = batch["depth_extras"].to(self.device, non_blocking=True)
        gt_future   = batch["traj"].to(self.device, non_blocking=True)
        last_pos    = batch["last_pos"].to(self.device, non_blocking=True)

        traj_res_k, mode_logits = self.forward(frames, depth_extras, init_masks=init_masks, init_labels=init_labels)
        
        pred_abs_k = self.to_abs(traj_res_k, last_pos)
        ade, fde, loss = best_of_k_loss(pred_abs_k, mode_logits, gt_future)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        mr = metrics_best_of_k(pred_abs_k, gt_future, r=self.mr_radius)["MR@2m"]
        m = {"ADE": ade.item(), "FDE": fde.item(), "mADE": ade.item(), "mFDE": fde.item(), "MR@2m": mr, "loss": loss.item()}
        return m, pred_abs_k.detach()

    def validation_step(self, batch):
        self.eval()
        with torch.inference_mode():
            frames      = batch["frames"].to(self.device, non_blocking=True)
            init_masks  = batch["init_masks"]
            init_labels = batch["init_labels"]
            depth_extras = batch["depth_extras"].to(self.device, non_blocking=True) 
            gt_future   = batch["traj"].to(self.device, non_blocking=True)
            last_pos    = batch["last_pos"].to(self.device, non_blocking=True)

            traj_res_k, mode_logits = self.forward(frames, depth_extras, init_masks=init_masks, init_labels=init_labels)
            pred_abs_k = self.to_abs(traj_res_k, last_pos)
            m = metrics_best_of_k(pred_abs_k, gt_future, r=self.mr_radius)

    
            return m, pred_abs_k