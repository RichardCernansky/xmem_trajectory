# traj/predictor.py
import sys
from typing import List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.inference_core import InferenceCore
from traj.early_fusion import EarlyFusionAdapter


class XMemMMBackboneWrapper(nn.Module):
    def __init__(self,
                 mm_cfg: Dict,
                 xmem,
                 device: str = "cuda",
                 n_lidar: int = 0,
                 fusion_mode: str = "concat",
                 use_bn: bool = False):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net = xmem
        self.mm_cfg = mm_cfg
        self.hidden_dim = getattr(self.net, "hidden_dim", mm_cfg.get("hidden_dim", 256))
        self.fusion = EarlyFusionAdapter(
            n_lidar=n_lidar, mode=fusion_mode, out_channels=3, use_bn=use_bn
        ) if n_lidar > 0 else nn.Identity()
        if isinstance(self.fusion, nn.Module):
            self.fusion.to(self.device)

        # hook scratch
        self._hidden_per_seq: List[Optional[torch.Tensor]] = []
        self._hook_target_idx: Optional[int] = None
        self._hook = None
        self._register_hidden_hook()

        # engines/streams (one per batch item)
        self.engines: List[InferenceCore] = []
        self.streams: List[Optional[torch.cuda.Stream]] = []

    # ---------------- hooks ----------------
    def _register_hidden_hook(self):
        # decoder returns (hidden_state, logits); we grab hidden_state which should be [1,K,C,H,W]
        def _grab_hidden(module, inputs, output):
            if self._hook_target_idx is not None and self._hook_target_idx < len(self._hidden_per_seq):
                self._hidden_per_seq[self._hook_target_idx] = output[0]
        if self._hook is not None:
            self._hook.remove()
        self._hook = self.net.decoder.register_forward_hook(_grab_hidden)

    @staticmethod
    def _masked_avg_pool_single(hid: torch.Tensor,
                                prob: Optional[torch.Tensor],
                                ch: int = 0,
                                eps: float = 1e-6) -> torch.Tensor:
        """
        Soft masked average pooling for a single object/channel.
        - hid:  [1,C,H,W]  OR  [1,K,C,H,W]
        - prob: [1,K,Hm,Wm] OR [K,Hm,Wm]  (NO background channel). If None -> plain mean pool.
        - ch:   target object's channel index (when K>1). For K=1 use ch=0.
        Returns: [C]
        """
        # select the hidden slice
        if hid.dim() == 5:
            # [1,K,C,H,W] -> pick target channel
            hid_sel = hid[:, ch]                        # [1,C,H,W]
        elif hid.dim() == 4:
            # shared map
            hid_sel = hid                               # [1,C,H,W]
        else:
            raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

        # no mask available -> plain mean over H,W
        if prob is None:
            return hid_sel.mean(dim=(2, 3)).squeeze(0)  # [C]

        # normalize prob shape to [1,K,Hm,Wm]
        if prob.dim() == 3:
            prob = prob.unsqueeze(0)
        elif prob.dim() != 4:
            raise RuntimeError(f"Unexpected prob shape: {tuple(prob.shape)}")

        # pick target mask
        if ch >= prob.size(1):
            # safety: if ch out of range, fall back to mean
            return hid_sel.mean(dim=(2, 3)).squeeze(0)
        m = prob[:, ch:ch+1]                            # [1,1,Hm,Wm]

        # device/dtype + resize to features
        if m.device != hid_sel.device:
            m = m.to(hid_sel.device)
        if m.dtype != hid_sel.dtype:
            m = m.to(hid_sel.dtype)
        if m.shape[-2:] != hid_sel.shape[-2:]:
            m = F.interpolate(m, size=hid_sel.shape[-2:], mode="bilinear", align_corners=False)

        # masked average pooling
        num = (hid_sel * m).sum(dim=(2, 3))            # [1,C]
        den = m.sum(dim=(2, 3)).clamp_min(eps)         # [1,1]
        out = (num / den).squeeze(0)                   # [C]

        # last safety against NaN/Inf (e.g., empty mask after resize)
        if not torch.isfinite(out).all():
            return hid_sel.mean(dim=(2, 3)).squeeze(0)
        return out


    # ---------------- engine mgmt ----------------
    def _ensure_capacity(self, B: int):
        cur = len(self.engines)
        if cur >= B:
            return
        need = B - cur
        self.engines += [InferenceCore(self.net, config=self.mm_cfg) for _ in range(need)]
        if torch.cuda.is_available() and self.device.type == "cuda":
            dev = self.device
            self.streams += [torch.cuda.Stream(device=dev) for _ in range(need)]
        else:
            self.streams += [None for _ in range(need)]

    def _hard_reset_engine(self, eng: InferenceCore):
        if hasattr(eng, "clear_memory"):
            eng.clear_memory()
        if hasattr(eng, "all_labels"):
            eng.all_labels = None
        if hasattr(eng, "ti"):
            eng.ti = 0
        if hasattr(eng, "curr_ti"):
            eng.curr_ti = 0
        if hasattr(eng, "last_mem_ti"):
            eng.last_mem_ti = -10_000

    # ---------------- forward ----------------
    @torch.no_grad()
    def forward(
        self,
        frames: torch.Tensor,                      # [B,T,3,H,W]
        init_masks: Optional[List[torch.Tensor]],  # len B, each [K_i,H,W] or [H,W] or None
        init_labels: Optional[List[List[int]]],    # len B, each K_i labels
        lidar_maps: Optional[torch.Tensor] = None  # [B,T,C_lidar,H,W] or None
    ) -> torch.Tensor:
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {frames.shape}"
        B, T, _, H, W = frames.shape
        frames = frames.to(self.device, non_blocking=True)
        has_lidar = lidar_maps is not None
        if has_lidar:
            lidar_maps = lidar_maps.to(self.device, non_blocking=True)

        if init_masks is not None:
            assert isinstance(init_masks, list) and len(init_masks) == B
        if init_labels is not None:
            assert isinstance(init_labels, list) and len(init_labels) == B

        self._ensure_capacity(B)
        for eng in self.engines[:B]:
            self._hard_reset_engine(eng)
        self._hidden_per_seq = [None for _ in range(B)]

        # t=0 masks/labels
        masks0: List[torch.Tensor] = []
        labels0: List[List[int]] = []
        for b in range(B):
            m0 = None
            lab0 = None
            if init_masks is not None:
                m0 = init_masks[b]
                if m0 is not None and not torch.is_tensor(m0):
                    m0 = torch.as_tensor(m0)
                if m0 is not None and m0.ndim == 2:
                    m0 = m0.unsqueeze(0)          # -> [K0,H,W] with K0=1
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

        feats: List[List[torch.Tensor]] = [[] for _ in range(B)]
        target_label = [labels0[b][0] for b in range(B)]  # choose first as target by default

        def _engine_step(eng, img, masks, labels, is_last):
            m_in = masks if masks is not None else None
            return eng.step(img, m_in, labels, end=is_last)

        for t in range(T):
            for b in range(B):
                eng = self.engines[b]
                stream = self.streams[b]
                rgb = frames[b, t]
                img = self.fusion(rgb.unsqueeze(0), lidar_maps[b, t].unsqueeze(0)).squeeze(0) if has_lidar else rgb
                self._hook_target_idx = b

                # GPU fast-path with per-seq stream; else run on current context
                if torch.cuda.is_available() and self.device.type == "cuda" and stream is not None:
                    with torch.cuda.stream(stream):
                        self._hidden_per_seq[b] = None
                        if t == 0:
                            if hasattr(eng, "set_all_labels"):
                                eng.set_all_labels(labels0[b])
                            _engine_step(eng, img, masks0[b], labels0[b], is_last=(t == T - 1))
                        else:
                            _engine_step(eng, img, None, None, is_last=(t == T - 1))
                        hid = self._hidden_per_seq[b]
                        prob = getattr(eng, "prob", None)
                        if hid is None:
                            continue  # usually only at t=0
                        if prob is not None:
                            feat = self._masked_avg_pool_single(hid, prob, ch=0)   # weighted by the target’s soft mask
                        else:
                            feat = self._masked_avg_pool_single(hid,prob,  ch=0)               # plain mean over H,W (no mask available)

                        feats[b].append(feat)  # [C]          
                else:
                    print("Error: GPU not accessible")

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()

        # stack to [B, T_feat, D] with D inferred from the first non-empty list
        sizes = [len(x) for x in feats]
        max_Tf = max(sizes) if sizes else 0
        # infer D from the first available feature
        D = None
        for b in range(B):
            if feats[b]:
                D = feats[b][0].numel()
                break
        if D is None:
            D = self.hidden_dim
        out = torch.zeros(B, max_Tf, D, device=self.device)
        for b in range(B):
            if feats[b]:
                fb = torch.stack(feats[b], dim=0)  # [Tb, C]
                out[b, :fb.size(0)] = fb
        return out
