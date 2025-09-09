# traj/predictor_parallel.py
import sys
from typing import List, Optional, Dict
import torch
import torch.nn as nn

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.inference_core import InferenceCore

class XMemMMBackboneWrapper(nn.Module):
    """
    Parallel over sequences (batch); sequential over time (t) per sequence.
    Uses one InferenceCore and one CUDA stream per batch item.
    """
    def __init__(self, mm_cfg: Dict, xmem, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.net = xmem
        self.mm_cfg = mm_cfg
        self.hidden_dim = getattr(self.net, "hidden_dim", mm_cfg.get("hidden_dim", 256))

        # per-forward-call scratch
        self._hidden_per_seq: List[Optional[torch.Tensor]] = []
        self._hook_target_idx: Optional[int] = None

        # shared hook on decoder; we route to the active seq via _hook_target_idx
        self._hook = None
        self._register_hidden_hook()

        # lazily grown pools
        self.engines: List[InferenceCore] = []
        self.streams: List[Optional[torch.cuda.Stream]] = []

    def _register_hidden_hook(self):
        def _grab_hidden(module, inputs, output):
            # output[0] = hidden_state
            if self._hook_target_idx is not None and self._hook_target_idx < len(self._hidden_per_seq):
                self._hidden_per_seq[self._hook_target_idx] = output[0]
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

    def _ensure_capacity(self, B: int):
        cur = len(self.engines)
        if cur >= B:
            return
        need = B - cur
        self.engines += [InferenceCore(self.net, config=self.mm_cfg) for _ in range(need)]
        if torch.cuda.is_available():
            self.streams += [torch.cuda.Stream(device=self.device) for _ in range(need)]
        else:
            self.streams += [None for _ in range(need)]

    @torch.no_grad()
    def forward(
        self,
        frames: torch.Tensor,                      # [B,T,3,H,W]
        init_masks: Optional[List[torch.Tensor]],  # len B, each [K_i,H,W] or [H,W]
        init_labels: Optional[List[List[int]]],    # len B, each K_i labels
        **_,
    ) -> torch.Tensor:
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {frames.shape}"
        B, T, C, H, W = frames.shape
        frames = frames.to(self.device, non_blocking=True)

        if init_masks is not None:
            assert isinstance(init_masks, list) and len(init_masks) == B
        if init_labels is not None:
            assert isinstance(init_labels, list) and len(init_labels) == B

        self._ensure_capacity(B)

        # reset engines + scratch
        for eng in self.engines[:B]:
            eng.clear_memory()
            eng.all_labels = None
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
                m0 = torch.ones(1, H, W, dtype=frames.dtype)
                lab0 = [1]
            if lab0 is None or len(lab0) != int(m0.size(0)):
                lab0 = list(range(1, int(m0.size(0)) + 1))
            masks0.append(m0.to(self.device, dtype=torch.float32, non_blocking=True))
            labels0.append(lab0)

        # collect features per sequence
        feats: List[List[torch.Tensor]] = [[] for _ in range(B)]

        # sequential in t, overlapped across sequences via CUDA streams
        for t in range(T):
            for b in range(B):
                eng = self.engines[b]
                stream = self.streams[b]
                rgb = frames[b, t]  # [3,H,W]

                # route hook outputs to the right slot
                self._hook_target_idx = b

                if torch.cuda.is_available():
                    with torch.cuda.stream(stream):
                        self._hidden_per_seq[b] = None
                        if t == 0:
                            eng.set_all_labels(labels0[b])
                            eng.step(rgb, masks0[b], labels0[b], end=(t == T - 1))
                        else:
                            eng.step(rgb, None, None, end=(t == T - 1))
                        hid = self._hidden_per_seq[b]
                        if hid is not None:
                            feats[b].append(self._pool_hidden(hid))
                else:
                    # CPU fallback (no real overlap)
                    self._hidden_per_seq[b] = None
                    if t == 0:
                        eng.set_all_labels(labels0[b])
                        eng.step(rgb, masks0[b], labels0[b], end=(t == T - 1))
                    else:
                        eng.step(rgb, None, None, end=(t == T - 1))
                    hid = self._hidden_per_seq[b]
                    if hid is not None:
                        feats[b].append(self._pool_hidden(hid))

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure all streams finished

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
