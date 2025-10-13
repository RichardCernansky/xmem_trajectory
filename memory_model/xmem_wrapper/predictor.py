import sys
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.configs.filenames import XMEM_CHECKPOINT, TRAIN_CONFIG, XMEM_CONFIG
from trainer.utils import open_config

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.memory_manager import MemoryManager
from model.aggregate import aggregate
from util.tensor_util import pad_divide_by
from model.network import XMem

def load_xmem(backbone_ckpt=XMEM_CHECKPOINT, device="cuda"):
    cfg = {"single_object": False}
    net = XMem(cfg, model_path=backbone_ckpt, map_location="cpu")
    state = torch.load(backbone_ckpt, map_location="cpu")
    net.load_weights(state, init_as_zero_if_needed=True)
    net.to(device)
    return net

# to do  - final feats graph will probably be corrupted bcause of inplace ops
# detach union?
class XMemBackboneWrapper(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.train_config = open_config(TRAIN_CONFIG)
        self.xmem_config = open_config(XMEM_CONFIG)
        self.xmem = load_xmem()
        self.xmem_core = self.xmem.to(self.device)
        self.hidden_dim = getattr(self.xmem_core, "hidden_dim")
        self.xmem_config["hidden_dim"] = self.hidden_dim

        self.mem_every = self.xmem_config["mem_every"]
        self.deep_update_every = self.xmem_config["deep_update_every"]
        self.enable_long_term = self.xmem_config["enable_long_term"]
        self.deep_update_sync = (self.deep_update_every < 0)

        # freeze all parameters
        for p in self.xmem_core.parameters():
            p.requires_grad = False

        # unfreeze decoder if wanted
        if self.train_config["train_xmem_decoder"] == True:
            for p in self.xmem_core.decoder.parameters():
                p.requires_grad = True


    def train(self, mode: bool = True):
        super().train(mode)
        key_enc = self.xmem_core.key_encoder
        val_enc = self.xmem_core.value_encoder
        dec = self.xmem_core.decoder

        if key_enc is not None:
            key_enc.train(mode) if self.train_config.get("train_xmem_key_encoder") else key_enc.eval()
        if val_enc is not None:
            val_enc.train(mode) if self.train_config.get("train_xmem_val_encoder") else val_enc.eval()
        if dec is not None:
            dec.train(mode) if self.train_config.get("train_xmem_decoder") else dec.eval()
        return self


    def forward(self, frames_cam, frames_lidar, *, init_masks=None):
        """
        
        - Write on t < t_in - 1.
        - At t=0: write with provided TARGET mask (GT).
        - For t>0: write with XMem's predicted mask at that step.
        Returns: (B,T,D) pooled timestep features.
        """
        B, T, _, H, W = frames_cam.shape
        dev = self.device
        frames_cam   = frames_cam.to(dev, non_blocking=True)
        frames_lidar = frames_lidar.to(dev, non_blocking=True)

        all_seq_feats = []

        for b in range(B):
            mm = MemoryManager(config=self.xmem_config.copy()); mm.ti = -1; mm.set_hidden(None)

            # --- t=0 target mask (required once). If none, fallback to ones. ---
            m0 = None if init_masks is None else init_masks[b]
            if m0 is not None and not torch.is_tensor(m0): m0 = torch.as_tensor(m0)
            if m0 is not None and m0.ndim == 2: m0 = m0.unsqueeze(0)      # (1,H,W)
            if m0 is None:
                m0 = torch.ones(1, H, W, dtype=torch.float32, device=dev) # fallback
            lab0 = [1]                                                    # single-object
            mm.all_labels = lab0

            seq_feats = []
            have_memory = False  # becomes True after we first write

            for t in range(T):
                mm.ti += 1
                write_this = (t < T - 1)  # write every frame except last

                # --- camera key + decoder pyramid (for WRITE + DECODE) ---
                img_cam,_ = pad_divide_by(frames_cam[b, t], 16); img_cam = img_cam.unsqueeze(0)
                _, _, _, f16c, f8c, f4c = self.xmem_core.encode_key(
                    img_cam, need_ek=False, need_sk=False
                )
                multi_cam = (f16c, f8c, f4c)

                # --- LiDAR key (for READ) ---
                img_lid,_ = pad_divide_by(frames_lidar[b, t], 16); img_lid = img_lid.unsqueeze(0)
                k_lid, sh_lid, sel_lid, _, _, _ = self.xmem_core.encode_key(
                    img_lid, need_ek=True, need_sk=write_this
                )

                # --- READ (only if we already have memory) & get XMem mask ---
                pred_prob_with_bg = None
                hidden_local = None
                if have_memory:
                    mem_rd = mm.match_memory(k_lid, sel_lid).unsqueeze(0)
                    hidden_local, _, pred_prob_with_bg = self.xmem_core.segment(
                        multi_cam, mem_rd, mm.get_hidden(), h_out=True, strip_bg=False
                    )
                    pred_prob_with_bg = pred_prob_with_bg[0]  # (1+K, Hs, Ws)

                # --- Build WRITE mask ---
                to_write = None
                if write_this:
                    if t == 0:
                        # use GT target mask at t=0
                        m_pad,_ = pad_divide_by(m0.float(), 16)       # (1,Hs,Ws)
                        to_write = aggregate(m_pad, dim=0)            # (1+K, Hs, Ws) with K=1
                    else:
                        # use XMem-predicted mask at t>0 (if available)
                        if pred_prob_with_bg is not None and pred_prob_with_bg.shape[0] > 1:
                            to_write = pred_prob_with_bg.detach()     # already (1+K,Hs,Ws)
                        else:
                            # fallback if memory not yet written (shouldn't happen after t=0)
                            m_pad,_ = pad_divide_by(m0.float(), 16)
                            to_write = aggregate(m_pad, dim=0)

                # --- WRITE: camera (key,value) with the chosen mask ---
                if to_write is not None:
                    # Expect K=1; ensure hidden state matches
                    h_cur = mm.get_hidden()
                    need_init = (h_cur is None) or (getattr(h_cur, "shape", None) is not None and h_cur.shape[1] != (to_write.shape[0]-1))
                    if need_init:
                        mm.create_hidden_state(int(to_write.shape[0]-1), k_lid); h_cur = mm.get_hidden()

                    v_cam, h2 = self.xmem_core.encode_value(
                        img_cam, f16c, h_cur, to_write[1:].unsqueeze(0), is_deep_update=False
                    )
                    mm.add_memory(
                        k_lid, sh_lid, v_cam, lab0,
                        selection=sel_lid if self.xmem_config["enable_long_term"] else None
                    )
                    mm.set_hidden(h2)
                    have_memory = True

                # --- FEATURE for this timestep ---
                if hidden_local is not None:
                    # Prefer masked pooling with current XMem union mask; fallback to mean
                    if pred_prob_with_bg is not None:
                        if pred_prob_with_bg.shape[0] > 1:
                            # union over K objects (K=1 in your case)
                            union = pred_prob_with_bg[1:].max(dim=0)[0].unsqueeze(0).unsqueeze(0)  # (1,1,Hs,Ws)
                            if union.shape[-2:] != hidden_local.shape[-2:]:
                                union = F.interpolate(union, size=hidden_local.shape[-2:], mode="bilinear", align_corners=False)
                            num = (hidden_local * union).sum(dim=(2,3))
                            den = union.sum(dim=(2,3)).clamp_min(1e-6)
                            feat = (num / den).squeeze(0)    # [C]
                        else:
                            feat = hidden_local.mean(dim=(2,3)).squeeze(0)
                    else:
                        feat = hidden_local.mean(dim=(2,3)).squeeze(0)
                else:
                    # At t=0 before first read, use key as a crude feature
                    feat = k_lid.mean(dim=(2,3)).squeeze(0)           # [D_key]

                seq_feats.append(feat)

            all_seq_feats.append(torch.stack(seq_feats, dim=0))  # (T,D)

        return torch.stack(all_seq_feats, dim=0)  # (B,T,D)