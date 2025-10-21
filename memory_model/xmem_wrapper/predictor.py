import sys
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.configs.filenames import XMEM_CHECKPOINT, TRAIN_CONFIG, XMEM_CONFIG, REPO_ROOT
from trainer.utils import open_config

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

def pad_to_same16(img_cam, img_lid):
    _, H, W = img_cam.shape
    H16 = (H + 15) // 16 * 16
    W16 = (W + 15) // 16 * 16
    def pad(img):
        dH, dW = H16 - img.shape[-2], W16 - img.shape[-1]
        return F.pad(img, (0, dW, 0, dH))  # pad right/bottom
    return pad(img_cam), pad(img_lid)

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

    def reset_memory(self, B: int):
        """Reset memory managers for a new batch of B sequences."""
        self.mms = []
        for _ in range(B):
            mm = MemoryManager(config=self.xmem_config.copy())
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
        if hid.dim() == 5:   # [1,K,C,H,W]
            hid_sel = hid[:, ch-1]
        elif hid.dim() == 4: # [1,C,H,W]
            hid_sel = hid
        else:
            raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

        if prob_with_bg is None:
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        if prob_with_bg.dim() == 3:
            prob_with_bg = prob_with_bg.unsqueeze(0)  # [1,K,Hm, Wm]
        if ch >= prob_with_bg.size(1):
            return hid_sel.mean(dim=(2, 3)).squeeze(0)

        m = prob_with_bg[:, ch:ch+1]
        if m.shape[-2:] != hid_sel.shape[-2:]:
            m = F.interpolate(m, size=hid_sel.shape[-2:], mode="bilinear", align_corners=False)
        num = (hid_sel * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp_min(eps)
        out = (num / den).squeeze(0)
        if not torch.isfinite(out).all():
            return hid_sel.mean(dim=(2, 3)).squeeze(0)
        return out

    def forward_step(self, t: int, frames_cam_t, frames_lidar_t, *, init_masks, init_labels):
        dev = self.device
        B, _, H, W = frames_cam_t.shape
        if t == 0:
            self.reset_memory(B)
        feats_out = []

        for b in range(B):
            mm = self.mms[b]
            mm.ti = t
            m0 = init_masks[b].to(dev)
            lab0 = init_labels[b]
            if isinstance(lab0, torch.Tensor):
                lab0 = [int(x) for x in lab0.flatten().tolist()]
            elif isinstance(lab0, (list, tuple)):
                lab0 = list(map(int, lab0))
            else:
                lab0 = [int(lab0)]
            mm.all_labels = lab0

            img_cam, img_lid = pad_to_same16(frames_cam_t[b], frames_lidar_t[b])
            k_cam, sh_cam, sel_cam, f16c, f8c, f4c = self.xmem_core.encode_key(
                img_cam.unsqueeze(0), need_ek=True, need_sk=True
            )
            k_lid, sh_lid, sel_lid, f16l, f8l, f4l = self.xmem_core.encode_key(
                img_lid.unsqueeze(0), need_ek=True, need_sk=True
            )
            multi_cam = (f16c, f8c, f4c)

            if t < 5:
                m_pad, _ = pad_divide_by(m0.float(), 16)
                to_write = aggregate(m_pad, dim=0)  # (1+K,H,W)
                K_write = int(to_write.shape[0] - 1)
                if K_write > 0:
                    h_cur = mm.get_hidden()
                    if (h_cur is None) or (h_cur.shape[1] != K_write):
                        mm.create_hidden_state(K_write, k_lid)
                        h_cur = mm.get_hidden()

                    v_cam, h2 = self.xmem_core.encode_value(
                        img_cam.unsqueeze(0), f16c, h_cur,
                        to_write[1:].unsqueeze(0),
                        is_deep_update=False
                    )

                    bsz, Kc, Cc, Hc, Wc = v_cam.shape
                    Hs_k, Ws_k = k_lid.shape[-2:]
                    if (Hc, Wc) != (Hs_k, Ws_k):
                        v4 = v_cam.view(bsz, Kc * Cc, Hc, Wc)
                        v4 = F.interpolate(v4, size=(Hs_k, Ws_k), mode="bilinear", align_corners=False)
                        v_cam = v4.view(bsz, Kc, Cc, Hs_k, Ws_k)

                    assert v_cam.shape[1] == K_write, f"K mismatch: {v_cam.shape[1]} vs {K_write}"
                    obj_ids = list(range(1, K_write + 1))
                    mm.add_memory(k_lid, sh_lid, v_cam, obj_ids, selection=sel_lid)
                    mm.set_hidden(h2)

                self.have_memory[b] = (mm.work_mem.key is not None) and (mm.work_mem.size > 0)
                feats_out.append(None)
                continue


            have_real = (mm.work_mem.key is not None) and (mm.work_mem.size > 0)
            hidden_local, pred_prob_with_bg = None, None
            if have_real:
                mem_rd = mm.match_memory(k_lid, sel_lid if self.enable_long_term else None).unsqueeze(0)
                hidden_local, _, pred_prob_with_bg = self.xmem_core.segment(
                    multi_cam, mem_rd, mm.get_hidden(), h_out=True, strip_bg=False
                )
                pred_prob_with_bg = pred_prob_with_bg[0]

            do_write = (self.mem_every > 0 and t % self.mem_every == 0)
            if do_write:
                if pred_prob_with_bg is not None and pred_prob_with_bg.shape[0] > 1:
                    to_write = pred_prob_with_bg.detach()
                else:
                    m_pad, _ = pad_divide_by(m0.float(), 16)
                    to_write = aggregate(m_pad, dim=0)
                K_write = int(to_write.shape[0] - 1)
                if K_write > 0:
                    h_cur = mm.get_hidden()
                    if (h_cur is None) or (h_cur.shape[1] != K_write):
                        mm.create_hidden_state(K_write, k_lid)
                        h_cur = mm.get_hidden()

                    v_cam, h2 = self.xmem_core.encode_value(
                        img_cam.unsqueeze(0), f16c, h_cur,
                        to_write[1:].unsqueeze(0),
                        is_deep_update=(self.deep_update_every > 0 and t % self.deep_update_every == 0)
                    )

                    bsz, Kc, Cc, Hc, Wc = v_cam.shape
                    Hs_k, Ws_k = k_lid.shape[-2:]
                    if (Hc, Wc) != (Hs_k, Ws_k):
                        v4 = v_cam.view(bsz, Kc * Cc, Hc, Wc)
                        v4 = F.interpolate(v4, size=(Hs_k, Ws_k), mode="bilinear", align_corners=False)
                        v_cam = v4.view(bsz, Kc, Cc, Hs_k, Ws_k)

                    assert v_cam.shape[1] == K_write, f"K mismatch: {v_cam.shape[1]} vs {K_write}"
                    obj_ids = list(range(1, K_write + 1))
                    mm.add_memory(k_lid, sh_lid, v_cam, obj_ids, selection=sel_lid)
                    mm.set_hidden(h2)

            self.have_memory[b] = (mm.work_mem.key is not None) and (mm.work_mem.size > 0)

            if hidden_local is None:
                feats_out.append(None)
                continue

            if pred_prob_with_bg is not None and pred_prob_with_bg.shape[0] > 1:
                feat = self._masked_avg_pool_single(hidden_local, pred_prob_with_bg, ch=1)
            else:
                feat = hidden_local.mean(dim=(2, 3)).squeeze(0)
            feats_out.append(feat)

        D = self.hidden_dim
        out_feats = torch.zeros(B, D, device=dev)
        for b in range(B):
            if feats_out[b] is not None:
                out_feats[b] = feats_out[b]
        return out_feats



