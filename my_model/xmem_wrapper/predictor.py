import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.configs.filenames import XMEM_CHECKPOINT, XMEM_CONFIG, REPO_ROOT
from trainer.utils import open_config

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.memory_manager import MemoryManager
from model.aggregate import aggregate
from util.tensor_util import pad_divide_by
from model.network import XMem


def load_xmem(device: torch.device, train_config):
    cfg = {"single_object": False}
    xmem_resume = bool(train_config.get("xmem_resume", False))
    if xmem_resume:
        ckpt_path = train_config.get("xmem_model", XMEM_CHECKPOINT)
        net = XMem(cfg, model_path=None, map_location="cpu")           # create XMem architecture
        state = torch.load(ckpt_path, map_location="cpu")              # load checkpoint to CPU
        net.load_weights(state, init_as_zero_if_needed=True)           # load weights into net
    else:
        print("NOT RESUMING XMEM")
        net = XMem(cfg, model_path=None, map_location="cpu")           # XMem from scratch (random init)
    net.to(device)                                                     # move to target device
    return net



class XMemBackboneWrapper(nn.Module):
    def __init__(self, device: str, train_config: dict):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self. train_config = train_config
        self.xmem_config = open_config(XMEM_CONFIG)

        self.xmem = load_xmem(self.device, self.train_config)
        self.xmem_core = self.xmem.to(self.device)

        self.hidden_dim = getattr(self.xmem_core, "hidden_dim")
        self.xmem_config["hidden_dim"] = self.hidden_dim

        self.mem_every = self.xmem_config["mem_every"]
        self.deep_update_every = self.xmem_config["deep_update_every"]
        self.enable_long_term = self.xmem_config["enable_long_term"]
        self.deep_update_sync = self.deep_update_every < 0

        for p in self.xmem_core.parameters():
            p.requires_grad = False                                     # freeze everything by default

        if self.train_config.get("train_xmem_decoder"):
            for p in self.xmem_core.decoder.parameters():
                p.requires_grad = True                                  # unfreeze decoder

        if self.train_config.get("train_xmem_key_encoder"):
            for p in self.xmem_core.key_encoder.parameters():
                p.requires_grad = True                                  # unfreeze key encoder

        if self.train_config.get("train_xmem_val_encoder"):
            for p in self.xmem_core.value_encoder.parameters():
                p.requires_grad = True                                  # unfreeze value encoder

    def reset_memory(self, B: int):
        self.mms = []
        for _ in range(B):
            mm = MemoryManager(config=self.xmem_config.copy())          # one MemoryManager per sequence in batch
            mm.ti = -1
            mm.set_hidden(None)
            self.mms.append(mm)
        self.have_memory = [False] * B

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
                                prob_with_bg: Optional[torch.Tensor],
                                ch: int = 1,
                                eps: float = 1e-6) -> torch.Tensor:
        if hid.dim() == 5:
            hid_sel = hid[:, ch - 1]                                    # select channel for object id ch
        elif hid.dim() == 4:
            hid_sel = hid
        else:
            raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

        if prob_with_bg is None:
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        if prob_with_bg.dim() == 3:
            prob_with_bg = prob_with_bg.unsqueeze(0)                    # [1, K, Hm, Wm]
        if ch >= prob_with_bg.size(1):
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        m = prob_with_bg[:, ch:ch + 1]                                  # select mask of object ch
        if m.shape[-2:] != hid_sel.shape[-2:]:
            m = F.interpolate(m, size=hid_sel.shape[-2:], mode="bilinear", align_corners=False)
        num = (hid_sel * m).sum(dim=(2, 3))                             # masked sum of features
        den = m.sum(dim=(2, 3)).clamp_min(eps)                          # mask area
        out = (num / den).squeeze(0)                                    # masked average pooling
        if not torch.isfinite(out).all():
            return hid_sel.mean(dim=(2, 3)).squeeze(0)
        return out

    def forward_step(self, t: int, frames_lidar_t, *, init_masks, init_labels):
        dev = self.device
        B = frames_lidar_t.size(0)
        if t == 0:
            self.reset_memory(B)

        feats_out = []
        writes_step = [None] * B
        mask_logits_out = []

        def _pad16_batch(x: torch.Tensor) -> torch.Tensor:
            H, W = x.shape[-2], x.shape[-1]
            H16 = (H + 15) // 16 * 16
            W16 = (W + 15) // 16 * 16
            if (H, W) == (H16, W16):
                return x
            return F.pad(x, (0, W16 - W, 0, H16 - H))

        def build_fg_logits(pred_prob_with_bg, m_current: torch.Tensor) -> torch.Tensor:
            Ht, Wt = m_current.shape[-2], m_current.shape[-1]

            if pred_prob_with_bg is not None:
                probs = pred_prob_with_bg
                if probs.dim() != 3:
                    raise RuntimeError(f"Unexpected pred_prob_with_bg shape: {probs.shape}")
                if probs.size(0) > 1:
                    prob_fg = probs[1:].max(dim=0).values
                else:
                    prob_fg = probs[0]
                prob_fg = prob_fg.unsqueeze(0)
                if prob_fg.shape[-2:] != (Ht, Wt):
                    prob_fg_4d = prob_fg.unsqueeze(0)
                    prob_fg_4d = F.interpolate(
                        prob_fg_4d, size=(Ht, Wt), mode="bilinear", align_corners=False
                    )
                    prob_fg = prob_fg_4d.squeeze(0)
                prob_fg = prob_fg.clamp(1e-5, 1 - 1e-5)
                logits_fg = torch.log(prob_fg / (1.0 - prob_fg))
                return logits_fg

            if m_current.dim() == 2:
                m_prob = m_current.unsqueeze(0).float()
            elif m_current.dim() == 3:
                if m_current.size(0) > 1:
                    m_prob = (m_current > 0).float().max(dim=0, keepdim=True).values
                else:
                    m_prob = m_current.float()
            else:
                raise RuntimeError(f"Unexpected init mask shape: {m_current.shape}")
            m_prob = m_prob.clamp(1e-5, 1 - 1e-5)
            logits_fg = torch.log(m_prob / (1.0 - m_prob))
            return logits_fg

        frames_lidar_t = _pad16_batch(frames_lidar_t)

        for b in range(B):
            mm = self.mms[b]
            mm.ti = t

            m_current = init_masks[b].to(dev)
            if m_current.dim() == 2:
                m_current = m_current.unsqueeze(0).unsqueeze(0)
            elif m_current.dim() == 3:
                m_current = m_current.unsqueeze(0)
            
            if m_current.shape[-2:] != (256, 256):
                m_current = F.interpolate(
                    m_current.float(),
                    size=(256, 256),
                    mode='nearest'
                )
            
            # Remove extra dims if needed: (1, 1, 256, 256) → (1, 256, 256)
            m_current = m_current.squeeze(0)  # Now shape: (1, 256, 256)

            lab0 = init_labels[b]
            if isinstance(lab0, torch.Tensor):
                lab0 = [int(x) for x in lab0.flatten().tolist()]
            elif isinstance(lab0, (list, tuple)):
                lab0 = list(map(int, lab0))
            else:
                lab0 = [int(lab0)]
            mm.all_labels = lab0

            k_l, sh_l, sel_l, f16l, f8l, f4l = self.xmem_core.encode_key(
                frames_lidar_t[b:b + 1], need_ek=True, need_sk=True
            )

            if sel_l.dim() == 2:
                sel_l = sel_l.unsqueeze(0)

            multi_lidar_b = (f16l, f8l, f4l)

            if t < 5:
                m_pad, _ = pad_divide_by(m_current.float(), 16)
                to_write = aggregate(m_pad, dim=0)
                K_write = int(to_write.shape[0] - 1)
                if K_write > 0:
                    h_cur = mm.get_hidden()
                    if (h_cur is None) or (h_cur.shape[1] != K_write):
                        mm.create_hidden_state(K_write, k_l)
                        h_cur = mm.get_hidden()

                    v_lid, h2 = self.xmem_core.encode_value(
                        frames_lidar_t[b:b + 1], f16l, h_cur,
                        to_write[1:].unsqueeze(0),
                        is_deep_update=False
                    )

                  

                    obj_ids = list(range(1, K_write + 1))
                    mm.add_memory(k_l, sh_l, v_lid, obj_ids, selection=sel_l)
                    mm.set_hidden(h2)

                self.have_memory[b] = (mm.work_mem.key is not None) and (mm.work_mem.size > 0)
                feats_out.append(None)
                logits_fg = build_fg_logits(None, m_current)
                mask_logits_out.append(logits_fg)
                continue

            have_real = (mm.work_mem.key is not None) and (mm.work_mem.size > 0)
            hidden_local, pred_prob_with_bg = None, None

            if have_real:
                mem_rd = mm.match_memory(
                    k_l, sel_l if self.enable_long_term else None
                ).unsqueeze(0)

                hidden_local, _, pred_prob_with_bg = self.xmem_core.segment(
                    multi_lidar_b, mem_rd, mm.get_hidden(), h_out=True, strip_bg=False
                )
                pred_prob_with_bg = pred_prob_with_bg[0]

            do_write = self.mem_every > 0 and t % self.mem_every == 0
            if do_write:
                if pred_prob_with_bg is not None and pred_prob_with_bg.shape[0] > 1:
                    to_write = pred_prob_with_bg.detach()
                    source = "pred"
                else:
                    m_pad, _ = pad_divide_by(m_current.float(), 16)
                    to_write = aggregate(m_pad, dim=0)
                    source = "gt"

                K_write = int(to_write.shape[0] - 1)
                if K_write > 0:
                    h_cur = mm.get_hidden()
                    if (h_cur is None) or (h_cur.shape[1] != K_write):
                        mm.create_hidden_state(K_write, k_l)
                        h_cur = mm.get_hidden()

                    v_lid, h2 = self.xmem_core.encode_value(
                        frames_lidar_t[b:b + 1], f16l, h_cur,
                        to_write[1:].unsqueeze(0),
                        is_deep_update=(self.deep_update_every > 0 and t % self.deep_update_every == 0)
                    )


                    obj_ids = list(range(1, K_write + 1))
                    mm.add_memory(k_l, sh_l, v_lid, obj_ids, selection=sel_l)
                    mm.set_hidden(h2)

                    writes_step[b] = {
                        "t": t,
                        "b": b,
                        "source": source,
                        "to_write": to_write.detach().float().cpu()
                    }

            self.have_memory[b] = (mm.work_mem.key is not None) and (mm.work_mem.size > 0)

            if hidden_local is None:
                feats_out.append(None)
            else:
                if pred_prob_with_bg is not None and pred_prob_with_bg.shape[0] > 1:
                    feat = self._masked_avg_pool_single(hidden_local, pred_prob_with_bg, ch=1)
                else:
                    feat = hidden_local.mean(dim=(2, 3)).squeeze(0)
                feats_out.append(feat)

            logits_fg = build_fg_logits(pred_prob_with_bg, m_current)
            mask_logits_out.append(logits_fg)

        D = self.hidden_dim
        tmpl = next((f for f in feats_out if f is not None), None)
        dtype = tmpl.dtype if tmpl is not None else torch.float32

        out_feats = torch.stack(
            [f if f is not None else torch.zeros(D, device=dev, dtype=dtype)
             for f in feats_out],
            dim=0
        )

        out_masks = torch.stack(mask_logits_out, dim=0)

        return out_feats, out_masks
 