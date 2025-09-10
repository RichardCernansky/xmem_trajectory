# traj/predictor.py
import sys
from typing import List, Optional, Dict
import torch
import torch.nn as nn

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.inference_core import InferenceCore
from traj.early_fusion import EarlyFusionAdapter


class XMemMMBackboneWrapper(nn.Module):
    """
    XMem backbone wrapper (hkchengrex/XMem compatible) with optional early fusion.
    - Parallel across batch (one engine per sequence), sequential over time per seq.
    - Exposes per-frame features via a decoder hidden-state hook, pooled to [C].
    """

    def __init__(self,
                 mm_cfg: Dict,
                 xmem,
                 device: str = "cuda",
                 n_lidar: int = 0,
                 fusion_mode: str = "concat",
                 use_bn: bool = False):
        super().__init__()
        # normalize device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # net (constructed outside with the same cfg) lives on device already
        self.net = xmem
        self.mm_cfg = mm_cfg
        self.hidden_dim = getattr(self.net, "hidden_dim", mm_cfg.get("hidden_dim", 256))

        # early fusion adapter (RGB [3] + LiDAR [C_lidar] -> fused [3])
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
        # decoder returns (hidden_state, logits) in this repo; grab hidden_state
        def _grab_hidden(module, inputs, output):
            if self._hook_target_idx is not None and self._hook_target_idx < len(self._hidden_per_seq):
                self._hidden_per_seq[self._hook_target_idx] = output[0]  # [B,C,H',W'] or [B,K,C,H',W']
        if self._hook is not None:
            self._hook.remove()
        self._hook = self.net.decoder.register_forward_hook(_grab_hidden)

    @staticmethod
    def _pool_hidden(hid: torch.Tensor) -> torch.Tensor:
        # [1,K,C,H',W'] -> [C]   or   [1,C,H',W'] -> [C]
        if hid.dim() == 5:
            return hid.mean(dim=(1, 3, 4)).squeeze(0)
        if hid.dim() == 4:
            return hid.mean(dim=(2, 3)).squeeze(0)
        raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

    # ---------------- engine mgmt ----------------
    def _ensure_capacity(self, B: int):
        cur = len(self.engines)
        if cur >= B:
            return
        need = B - cur
        # exact signature for hkchengrex/XMem
        self.engines += [InferenceCore(self.net, config=self.mm_cfg) for _ in range(need)]

        if torch.cuda.is_available() and self.device.type == "cuda":
            dev = self.device  # torch.device
            self.streams += [torch.cuda.Stream(device=dev) for _ in range(need)]
        else:
            self.streams += [None for _ in range(need)]

    def _hard_reset_engine(self, eng: InferenceCore):
        """Reset memory + common time counters across forks."""
        if hasattr(eng, "clear_memory"):
            eng.clear_memory()
        if hasattr(eng, "all_labels"):
            eng.all_labels = None
        # reset time indices if present (harmless if absent)
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

        # reset engines + scratch
        for eng in self.engines[:B]:
            self._hard_reset_engine(eng)
        self._hidden_per_seq = [None for _ in range(B)]

        # prepare t=0 masks/labels
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
                    m0 = m0.unsqueeze(0)
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

        # engine.step signature in hkchengrex/XMem:
        #   step(image: [3,H,W], mask: [K,H,W] or None, labels: list[int] or None, end: bool)
        def _engine_step(eng, img, masks, labels, is_last):
            m_in = masks if masks is not None else None
            return eng.step(img, m_in, labels, end=is_last)

        # sequential in t, overlapped across sequences via CUDA streams
        for t in range(T):
            for b in range(B):
                eng = self.engines[b]
                stream = self.streams[b]

                # Build the per-frame input
                rgb = frames[b, t]  # [3,H,W]
                if has_lidar:
                    lid = lidar_maps[b, t]  # [C_lidar,H,W]
                    # Early fusion expects BCHW; squeeze back for engine
                    fused = self.fusion(rgb.unsqueeze(0), lid.unsqueeze(0)).squeeze(0)  # [3,H,W]
                    img = fused
                else:
                    img = rgb  # [3,H,W]

                # route hook outputs to the right slot
                self._hook_target_idx = b

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
                        if hid is not None:
                            feats[b].append(self._pool_hidden(hid))
                else:
                    self._hidden_per_seq[b] = None
                    if t == 0:
                        if hasattr(eng, "set_all_labels"):
                            eng.set_all_labels(labels0[b])
                        _engine_step(eng, img, masks0[b], labels0[b], is_last=(t == T - 1))
                    else:
                        _engine_step(eng, img, None, None, is_last=(t == T - 1))
                    hid = self._hidden_per_seq[b]
                    if hid is not None:
                        feats[b].append(self._pool_hidden(hid))

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()

        # pad to [B, T_feat, D]
        sizes = [len(x) for x in feats]
        max_Tf = max(sizes) if sizes else 0
        D = self.hidden_dim
        out = torch.zeros(B, max_Tf, D, device=self.device)
        for b in range(B):
            if feats[b]:
                fb = torch.stack(feats[b], dim=0)  # [Tb, D]
                out[b, :fb.size(0)] = fb
        return out
