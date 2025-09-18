import sys
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.memory_manager import MemoryManager
from model.aggregate import aggregate
from util.tensor_util import pad_divide_by, unpad

from traj.early_fusion import EarlyFusionAdapter


class XMemMMBackboneWrapper(nn.Module):
    """
    End-to-end differentiable wrapper:
      - Implements XMem's memory policy locally (mem_every, deep_update_every, enable_long_term)
      - Calls XMem APIs directly (encode_key, segment, encode_value)
      - Keeps gradients for early fusion; optional fine-tune XMem via tune_xmem
    """
    def __init__(self,
                 mm_cfg: Dict,
                 xmem,
                 device: str = "cuda",
                 n_lidar: int = 0,
                 fusion_mode: str = "concat",
                 use_bn: bool = True,
                 tune_xmem: bool = False):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = xmem.to(self.device)

        # keep a local, editable copy of config
        self.mm_cfg = dict(mm_cfg)
        self.hidden_dim = getattr(self.net, "hidden_dim", self.mm_cfg.get("hidden_dim", 256))
        self.mm_cfg.setdefault("hidden_dim", self.hidden_dim)

        # memory policy
        self.mem_every = self.mm_cfg['mem_every']
        self.deep_update_every = self.mm_cfg['deep_update_every']
        self.enable_long_term = self.mm_cfg['enable_long_term']
        self.deep_update_sync = (self.deep_update_every < 0)

        # early fusion (learnable)
        self.fusion = EarlyFusionAdapter(
            n_lidar=n_lidar, mode=fusion_mode, out_channels=3, use_bn=use_bn
        ) if n_lidar > 0 else nn.Identity()
        if isinstance(self.fusion, nn.Module):
            self.fusion.to(self.device)

        # train/freeze mode for XMem
        self.tune_xmem = tune_xmem
        # self._set_xmem_mode()

    # in XMemMMBackboneWrapper
    def _set_xmem_mode(self):
        # Only set module mode; do NOT touch requires_grad here
        if self.tune_xmem:
            self.net.train()
        else:
            self.net.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_xmem_mode()
        return self

    @staticmethod
    def _masked_avg_pool_single(hid: torch.Tensor,
                                prob_no_bg: Optional[torch.Tensor],
                                ch: int = 0,
                                eps: float = 1e-6) -> torch.Tensor:
        """
        hid: [1,C,H,W] or [1,K,C,H,W]
        prob_no_bg: [K,Hm,Wm] or [1,K,Hm,Wm] (no background channel)
        returns: [C]
        """
        if hid.dim() == 5:
            hid_sel = hid[:, ch]  # [1,C,H,W]
        elif hid.dim() == 4:
            hid_sel = hid         # [1,C,H,W]
        else:
            raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

        if prob_no_bg is None:
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        if prob_no_bg.dim() == 3:
            prob_no_bg = prob_no_bg.unsqueeze(0)  # [1,K,Hm,Wm]
        if ch >= prob_no_bg.size(1):
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        m = prob_no_bg[:, ch:ch+1]  # [1,1,Hm,Wm]
        if m.shape[-2:] != hid_sel.shape[-2:]:
            m = F.interpolate(m, size=hid_sel.shape[-2:], mode="bilinear", align_corners=False)

        num = (hid_sel * m).sum(dim=(2, 3))       # [1,C]
        den = m.sum(dim=(2, 3)).clamp_min(eps)    # [1,1]
        out = (num / den).squeeze(0)              # [C]
        if not torch.isfinite(out).all():
            return hid_sel.mean(dim=(2, 3)).squeeze(0)
        return out

    def forward(
        self,
        frames: torch.Tensor,                      # [B,T,3,H,W]
        init_masks: Optional[List[torch.Tensor]],  # len B, each [K0,H,W] or [H,W] or None
        init_labels: Optional[List[List[int]]],    # len B, labels per object
        lidar_maps: Optional[torch.Tensor] = None  # [B,T,C_lidar,H,W] or None
    ) -> torch.Tensor:

        # inputs
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {frames.shape}"
        B, T, _, H, W = frames.shape
        frames = frames.to(self.device, non_blocking=True)
        has_lidar = lidar_maps is not None
        if has_lidar:
            lidar_maps = lidar_maps.to(self.device, non_blocking=True)

        # per-seq memory
        memories = [MemoryManager(config=self.mm_cfg.copy()) for _ in range(B)]
        curr_ti = [-1 for _ in range(B)]
        last_mem_ti = [0 for _ in range(B)]
        if not self.deep_update_sync:
            last_deep_update_ti = [-self.deep_update_every for _ in range(B)]
        else:
            last_deep_update_ti = [0 for _ in range(B)]

        # t=0 masks/labels
        masks0: List[torch.Tensor] = []
        labels0: List[List[int]] = []
        for b in range(B):
            m0 = None; lab0 = None
            if init_masks is not None:
                m0 = init_masks[b]
                if m0 is not None and not torch.is_tensor(m0):
                    m0 = torch.as_tensor(m0)
                if m0 is not None and m0.ndim == 2:
                    m0 = m0.unsqueeze(0)  # [K0,H,W]
            if init_labels is not None:
                lab0 = init_labels[b]
                if torch.is_tensor(lab0):
                    lab0 = [int(x) for x in lab0.tolist()]
        
            if m0 is None:
                m0 = torch.ones(1, H, W, dtype=frames.dtype, device=self.device)
                lab0 = [1]
            if lab0 is None or len(lab0) != int(m0.size(0)):
                lab0 = list(range(1, int(m0.size(0)) + 1))
            masks0.append(m0.to(self.device, dtype=torch.float32, non_blocking=True))
            labels0.append(lab0)
            memories[b].set_hidden(None)
            memories[b].ti = -1
            memories[b].all_labels = labels0[b]

        feats: List[List[torch.Tensor]] = [[] for _ in range(B)]

        # time loop
        for t in range(T):
            for b in range(B):
                curr_ti[b] += 1
                memories[b].ti = curr_ti[b]

                # policy flags (InferenceCore-like)
                is_mem_frame = ((curr_ti[b] - last_mem_ti[b] >= self.mem_every) or (t == 0)) and (t != T-1)
                need_segment = (t > 0)  # set True to segment every frame if desired
                is_deep_update = (
                    (self.deep_update_sync and is_mem_frame) or
                    (not self.deep_update_sync and (curr_ti[b] - last_deep_update_ti[b] >= self.deep_update_every))
                ) and (t != T-1)
                is_normal_update = (not self.deep_update_sync or not is_deep_update) and (t != T-1)

                # early fusion before XMem
                rgb = frames[b, t]  # [3,H,W]
                if has_lidar:
                    img = self.fusion(rgb.unsqueeze(0), lidar_maps[b, t].unsqueeze(0)).squeeze(0)
                else:
                    img = rgb

                # pad to /16
                img_pad, pad_meta = pad_divide_by(img, 16)
                img_pad = img_pad.unsqueeze(0)  # [1,3,H',W']

                # encode
                key, shrinkage, selection, f16, f8, f4 = self.net.encode_key(
                    img_pad,
                    need_ek=(self.enable_long_term or need_segment),
                    need_sk=is_mem_frame
                )
                multi = (f16, f8, f4)

                # segment (read) if needed
                pred_prob_with_bg = None
                prob_no_bg_for_pool = None
                hidden_local = None

                if need_segment:
                    memory_readout = memories[b].match_memory(key, selection).unsqueeze(0)
                    hidden_local, _, pred_prob_with_bg = self.net.segment(
                        multi,
                        memory_readout,
                        memories[b].get_hidden(),
                        h_out=True,          # ensure we have hidden for pooling
                        strip_bg=False
                    )
                    pred_prob_with_bg = pred_prob_with_bg[0]     # [1+K, Hs, Ws]
                    prob_no_bg_for_pool = pred_prob_with_bg[1:]  # [K, Hs, Ws]
                    if is_normal_update:
                        memories[b].set_hidden(hidden_local)

                # build write probs (separate tensor; do NOT mutate the pooled probs)
                if t == 0 and masks0[b] is not None:
                    # use GT mask to write initial memory; avoid any in-place on pred probs
                    m_pad, _ = pad_divide_by(masks0[b], 16)            # [K0, H', W']
                    if pred_prob_with_bg is not None:
                        m_pad = m_pad.to(dtype=pred_prob_with_bg.dtype, device=pred_prob_with_bg.device)
                    pred_prob_with_bg_for_write = aggregate(m_pad, dim=0)  # [1+K0, H', W']
                else:
                    pred_prob_with_bg_for_write = pred_prob_with_bg      # may be None if we skipped segment

                # write memory per policy; detach write mask to keep graph clean
                # ----- write memory if needed -----
                if is_mem_frame and pred_prob_with_bg_for_write is not None:
                    # how many objects (no BG)
                    K_write = int(pred_prob_with_bg_for_write.shape[0] - 1)

                    # ensure a valid hidden exists with the right K
                    h_cur = memories[b].get_hidden()
                    need_init = (h_cur is None) or (getattr(h_cur, "shape", None) is not None and h_cur.shape[1] != K_write)
                    if need_init:
                        memories[b].create_hidden_state(K_write, key)
                        h_cur = memories[b].get_hidden()

                    # NEVER backprop through the write mask; keep graph clean
                    write_obj = pred_prob_with_bg_for_write[1:].unsqueeze(0).detach()   # [1,K,H',W']

                    value, hidden2 = self.net.encode_value(
                        img_pad, f16, h_cur,
                        write_obj,
                        is_deep_update=is_deep_update
                    )
                    memories[b].add_memory(
                        key, shrinkage, value, labels0[b],
                        selection=selection if self.enable_long_term else None
                    )
                    last_mem_ti[b] = curr_ti[b]
                    if is_deep_update:
                        memories[b].set_hidden(hidden2)
                        last_deep_update_ti[b] = curr_ti[b]


                # feature for trajectory head (consistent 64-D)
                if (pred_prob_with_bg is not None) and isinstance(hidden_local, torch.Tensor):
                    print("using avg pool")
                    feat = self._masked_avg_pool_single(hidden_local, prob_no_bg_for_pool, ch=0)  # [C^h=64]
                else:
                    
                    feat = key.mean(dim=(2, 3)).squeeze(0)  # fallback: [C^k=64]
                feats[b].append(feat)

        # pad & stack to [B, T_feat, D] (no in-place writes)
        sizes = [len(x) for x in feats]
        max_Tf = max(sizes) if sizes else 0
        D = next((feats[b][0].shape[-1] for b in range(B) if feats[b]), self.hidden_dim)

        seqs = []
        for b in range(B):
            if feats[b]:
                fb = torch.stack(feats[b], dim=0)        # [Tb, D]
                pad = max_Tf - fb.size(0)
                if pad > 0:
                    fb = torch.cat([fb, fb.new_zeros(pad, fb.size(1))], dim=0)
            else:
                fb = torch.zeros(max_Tf, D, device=self.device)
            seqs.append(fb)
        out = torch.stack(seqs, dim=0)  # [B, max_Tf, D]
        return out




# If you fine-tune XMem, start by unfreezing only the decoder at a tiny LR (e.g., 1e-5).
# When writing memory, it’s common to detach the write mask—this stabilizes training and avoids huge graphs.
#PSEUDO
# for t in frames:
#   img = fuse(rgb, lidar)                 # learnable early fusion

#   key, f16,f8,f4 = encode_key(img)       # encoder + key head

#   if need_segment:                       # often True during training
#     readout = memory.match(key)
#     hidden, probs = segment((f16,f8,f4), readout, hidden_prev, h_out=True)
#     if is_normal_update:                 # per policy
#       hidden_prev = hidden               # update decoder hidden

#   # choose the mask used to WRITE memory (separate from pooling)
#   if t==0 and have_GT:
#     write_probs = build_from_GT_mask()   # aggregate(GT)
#   else:
#     write_probs = probs                  # predicted

#   if is_mem_frame and write_probs is not None:
#     ensure_hidden_initialized(K(write_probs))
#     encode_value(..., write_probs.detach())   # detach to keep graph safe
#     memory.add(key, value, ...)
#     if is_deep_update: hidden_prev = hidden2

#   # feature for trajectory head (always 64-D)
#   if probs is not None and hidden is not None:
#     feat = masked_avg_pool(hidden, probs_no_bg, ch=0)    # 64-D
#   else:
#     feat = key.mean(HW)                                  # 64-D fallback
