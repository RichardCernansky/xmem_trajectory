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
    net.to(device).eval()
    return net

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


    @staticmethod
    def _masked_avg_pool_single(hid: torch.Tensor,
                                prob_no_bg: Optional[torch.Tensor],
                                ch: int = 0,
                                eps: float = 1e-6) -> torch.Tensor:
        if hid.dim() == 5:   # [1,K,C,H,W]
            hid_sel = hid[:, ch]
        elif hid.dim() == 4: # [1,C,H,W]
            hid_sel = hid
        else:
            raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

        if prob_no_bg is None:
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        if prob_no_bg.dim() == 3:
            prob_no_bg = prob_no_bg.unsqueeze(0)  # [1,K,Hm, Wm]
        if ch >= prob_no_bg.size(1):
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        m = prob_no_bg[:, ch:ch+1]
        if m.shape[-2:] != hid_sel.shape[-2:]:
            m = F.interpolate(m, size=hid_sel.shape[-2:], mode="bilinear", align_corners=False)
        num = (hid_sel * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp_min(eps)
        out = (num / den).squeeze(0)
        if not torch.isfinite(out).all():
            return hid_sel.mean(dim=(2, 3)).squeeze(0)
        return out

    def forward(
        self,
        frames: torch.Tensor,                      # [B,T,3,H,W]
        init_masks: Optional[List[torch.Tensor]],  # len B, each [K0,H,W] or [H,W] or None
        init_labels: Optional[List[List[int]]],    # len B, labels per object
    ) -> torch.Tensor:
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {frames.shape}"
        B, T, _, H, W = frames.shape
        frames = frames.to(self.device, non_blocking=True)

        all_feats: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = [] 

        # --- process each sequence sequentially with a single MemoryManager ---
        for b in range(B):
            # per-sequence memory/state
            mm = MemoryManager(config=self.xmem_config.copy())
            mm.ti = -1
            mm.set_hidden(None)

            curr_ti = -1
            last_mem_ti = 0
            last_deep_update_ti = (0 if self.deep_update_sync else -self.deep_update_every)

            # prepare t=0 masks/labels
            if init_masks is not None:
                m0 = init_masks[b]
                if m0 is not None and not torch.is_tensor(m0):
                    m0 = torch.as_tensor(m0)
                if m0 is not None and m0.ndim == 2:
                    m0 = m0.unsqueeze(0)  # [K0,H,W]
            else:
                m0 = None

            if init_labels is not None:
                lab0 = init_labels[b]
                if torch.is_tensor(lab0):
                    lab0 = [int(x) for x in lab0.tolist()]
            else:
                lab0 = None

            if m0 is None:
                m0 = torch.ones(1, H, W, dtype=frames.dtype, device=self.device)
                lab0 = [1]
            if lab0 is None or len(lab0) != int(m0.size(0)):
                lab0 = list(range(1, int(m0.size(0)) + 1))

            m0 = m0.to(self.device, dtype=torch.float32, non_blocking=True)
            mm.all_labels = lab0

            seq_feats: List[torch.Tensor] = []
            seq_masks: List[torch.Tensor] = []

            for t in range(T):
                curr_ti += 1
                mm.ti = curr_ti

                is_mem_frame = ((curr_ti - last_mem_ti >= self.mem_every) or (t == 0)) and (t != T-1)
                need_segment = (t > 0)
                is_deep_update = (
                    (self.deep_update_sync and is_mem_frame) or
                    (not self.deep_update_sync and (curr_ti - last_deep_update_ti >= self.deep_update_every))
                ) and (t != T-1)
                is_normal_update = (not self.deep_update_sync or not is_deep_update) and (t != T-1)

                rgb = frames[b, t]                      # [3,H,W]
                # inside backbone forward, right after: rgb = frames[b, t]  # [3,H,W]
                # if rgb.dtype == torch.uint8:
                #     rgb = rgb.float() / 255.0
                # mean = torch.tensor([0.485,0.456,0.406], device=rgb.device)[:,None,None]
                # std  = torch.tensor([0.229,0.224,0.225], device=rgb.device)[:,None,None]
                # rgb = (rgb - mean) / std

                img_pad, _ = pad_divide_by(rgb, 16)
                img_pad = img_pad.unsqueeze(0)          # [1,3,H',W']

                key, shrinkage, selection, f16, f8, f4 = self.xmem_core.encode_key(
                    img_pad,
                    need_ek=(self.enable_long_term or need_segment),
                    need_sk=is_mem_frame
                )
                multi = (f16, f8, f4)

                pred_prob_with_bg = None
                prob_no_bg_for_pool = None
                hidden_local = None

                if need_segment:
                    memory_readout = mm.match_memory(key, selection).unsqueeze(0)
                    hidden_local, _, pred_prob_with_bg = self.xmem_core.segment(
                        multi,
                        memory_readout,
                        mm.get_hidden(),
                        h_out=True,
                        strip_bg=False
                    )
                    pred_prob_with_bg = pred_prob_with_bg[0]     # [1+K,Hs,Ws]
                    prob_no_bg_for_pool = pred_prob_with_bg[1:]  # [K,Hs,Ws]
                    if is_normal_update:
                        mm.set_hidden(hidden_local)
                
                 # --- build a union FG mask and upsample to (H, W) ---  ### NEW
                if pred_prob_with_bg is not None:
                    if pred_prob_with_bg.shape[0] > 1:
                        fg_small = pred_prob_with_bg[1:]              # [K, Hs, Ws]
                        union_small = fg_small.max(dim=0)[0]          # [Hs, Ws]
                    else:
                        union_small = torch.zeros_like(pred_prob_with_bg[0])
                    union_up = F.interpolate(
                        union_small.unsqueeze(0).unsqueeze(0),        # [1,1,Hs,Ws]
                        size=(H, W), mode="bilinear", align_corners=False
                    )[0, 0]                                           # [H, W]
                else:
                    union_up = torch.zeros(H, W, device=self.device)

                seq_masks.append(union_up.detach().cpu()) 

                # write probs (separate from pooled probs)
                if t == 0 and m0 is not None:
                    m_pad, _ = pad_divide_by(m0, 16)             # [K0,H',W']
                    if pred_prob_with_bg is not None:
                        m_pad = m_pad.to(dtype=pred_prob_with_bg.dtype, device=pred_prob_with_bg.device)
                    pred_prob_with_bg_for_write = aggregate(m_pad, dim=0)  # [1+K0,H',W']
                else:
                    pred_prob_with_bg_for_write = pred_prob_with_bg

                if is_mem_frame and pred_prob_with_bg_for_write is not None:
                    K_write = int(pred_prob_with_bg_for_write.shape[0] - 1)

                    h_cur = mm.get_hidden()
                    need_init = (h_cur is None) or (getattr(h_cur, "shape", None) is not None and h_cur.shape[1] != K_write)
                    if need_init:
                        mm.create_hidden_state(K_write, key)
                        h_cur = mm.get_hidden()

                    write_obj = pred_prob_with_bg_for_write[1:].unsqueeze(0).detach()
                    value, hidden2 = self.xmem_core.encode_value(
                        img_pad, f16, h_cur,
                        write_obj,
                        is_deep_update=is_deep_update
                    )
                    mm.add_memory(
                        key, shrinkage, value, lab0,
                        selection=selection if self.enable_long_term else None
                    )
                    last_mem_ti = curr_ti
                    if is_deep_update:
                        mm.set_hidden(hidden2)
                        last_deep_update_ti = curr_ti

                # 64-D feature for traj head
                if (pred_prob_with_bg is not None) and isinstance(hidden_local, torch.Tensor):
                    feat = self._masked_avg_pool_single(hidden_local, prob_no_bg_for_pool, ch=0)  # [64]
                else:
                    feat = key.mean(dim=(2, 3)).squeeze(0)  # [64]
                seq_feats.append(feat)
                
                
            all_masks.append(torch.stack(seq_masks, dim=0))
            all_feats.append(torch.stack(seq_feats, dim=0))  # [T, D]

        # stack to [B,T,D]
        max_T = max(x.size(0) for x in all_feats)
        D = all_feats[0].size(1)
        out = []
        for fb in all_feats:
            pad = max_T - fb.size(0)
            if pad > 0:
                fb = torch.cat([fb, fb.new_zeros(pad, D)], dim=0)
            out.append(fb)
    
        self.last_masks = torch.stack(all_masks, dim=0)               ### NEW  [B, T, H, W]
        return torch.stack(out, dim=0)  # [B,T,D]
