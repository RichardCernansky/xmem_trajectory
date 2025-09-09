# traj/predictor.py
import sys
from typing import List, Optional, Dict
import torch
import torch.nn as nn

REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.inference_core import InferenceCore


def xmem_mm_config(
    mem_every: int = 3,
    min_mid: int = 5, max_mid: int = 10,
    num_prototypes: int = 128,
    max_long_term: int = 10000,
    enable_long_term: bool = False,
    deep_update_every: int = 10**9,  # effectively disable deep update at t=0
    top_k: int = 30,
    single_object: bool = False,
    hidden_dim: int = 256,
) -> Dict:
    return {
        "mem_every": mem_every,
        "min_mid_term_frames": min_mid,
        "max_mid_term_frames": max_mid,
        "num_prototypes": num_prototypes,
        "max_long_term_elements": max_long_term,
        "enable_long_term": enable_long_term,
        "deep_update_every": deep_update_every,
        "top_k": top_k,
        "benchmark": False,
        "enable_long_term_count_usage": False,
        "single_object": single_object,
        "hidden_dim": hidden_dim,
    }


class XMemMMBackboneWrapper(nn.Module):
    """
    XMem backbone that delegates memory policy to InferenceCore.
    Exposes a per-frame feature vector by hooking the decoder's hidden state.
    Works with variable-K masks provided as LISTS per batch item.
    """

    def __init__(self, mm_cfg: Dict, xmem, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.net = xmem
        self.engine = InferenceCore(self.net, config=mm_cfg)
        self.last_hidden = None
        self._hook = None
        self._register_hidden_hook()

    def _register_hidden_hook(self):
        # decoder returns (hidden_state, logits); grab hidden_state on each step
        def _grab_hidden(module, inputs, output):
            self.last_hidden = output[0]  # [B, C, H', W'] or [B,K,C,H',W']
        if self._hook is not None:
            self._hook.remove()
        self._hook = self.net.decoder.register_forward_hook(_grab_hidden)

    @staticmethod
    def _pool_hidden(hid: torch.Tensor) -> torch.Tensor:
        """
        Accepts hidden of shape [1,K,C,H',W'] or [1,C,H',W'] and returns [C].
        """
        if hid.dim() == 5:   # [B(=1), K, C, H', W']
            return hid.mean(dim=(1, 3, 4)).squeeze(0)  # → [C]
        if hid.dim() == 4:   # [B(=1), C, H', W']
            return hid.mean(dim=(2, 3)).squeeze(0)     # → [C]
        raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

    @torch.no_grad()
    def forward(
        self,
        frames: torch.Tensor,                   # [B,T,3,H,W]
        init_masks: Optional[List[torch.Tensor]],   # list len B, each [K_i,H,W] or [H,W]
        init_labels: Optional[List[List[int]]],     # list len B, each K_i labels
        **_,
    ) -> torch.Tensor:
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {frames.shape}"
        B, T, C, H, W = frames.shape
        frames = frames.to(self.device)

        if init_masks is not None:
            assert isinstance(init_masks, list) and len(init_masks) == B
        if init_labels is not None:
            assert isinstance(init_labels, list) and len(init_labels) == B

        all_feats: List[torch.Tensor] = []

        for b in range(B):
            # reset per sequence
            self.engine.clear_memory()
            self.last_hidden = None
            self.engine.all_labels = None

            seq_feats: List[torch.Tensor] = []

            # prepare t=0 masks/labels (robust to None / shape issues)
            m0 = None
            lab0 = None
            if init_masks is not None:
                m0 = init_masks[b]
                if m0 is not None:
                    if not torch.is_tensor(m0):
                        m0 = torch.as_tensor(m0)
                    if m0.ndim == 2:      # [H,W] → [1,H,W]
                        m0 = m0.unsqueeze(0)
            if init_labels is not None:
                lab0 = init_labels[b]
                if torch.is_tensor(lab0):
                    lab0 = [int(x) for x in lab0.tolist()]

            # fallback if no masks
            if m0 is None:
                m0 = torch.ones(1, H, W, dtype=frames.dtype)  # CPU for now; moved to device below
                lab0 = [1]

            # sanity: align label count with K
            if lab0 is None or len(lab0) != int(m0.size(0)):
                lab0 = list(range(1, int(m0.size(0)) + 1))

            m0 = m0.to(self.device).float()

            for t in range(T):
                rgb = frames[b, t]  # [3,H,W]

                if t == 0:
                    # first frame initializes memory + labels
                    self.engine.set_all_labels(lab0)
                    _ = self.engine.step(rgb, m0, lab0, end=(t == T - 1))
                else:
                    _ = self.engine.step(rgb, None, None, end=(t == T - 1))

                if self.last_hidden is None:
                    continue  # usually only at t=0
                feat = self._pool_hidden(self.last_hidden)  # [C]
                seq_feats.append(feat)

            if seq_feats:
                all_feats.append(torch.stack(seq_feats, dim=0))  # [T_feat, C]
            else:
                D = getattr(self.net, "hidden_dim", 256)
                all_feats.append(torch.zeros(0, D, device=self.device))

        # pad time dimension across batch
        max_Tf = max(f.size(0) for f in all_feats)
        D = all_feats[0].size(-1) if max_Tf > 0 else getattr(self.net, "hidden_dim", 256)
        out = torch.zeros(B, max_Tf, D, device=self.device)
        for b, fb in enumerate(all_feats):
            if fb.numel():
                out[b, :fb.size(0)] = fb
        return out  # [B, T_feat, D]
