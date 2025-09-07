# --- standard library ---
import os
import sys

# --- third-party ---
import torch
import torch.nn as nn

# --- typing ---
from typing import List, Optional, Dict

# --- local repo ---
REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.inference_core import InferenceCore


# --- add this to build the InferenceCore config ---
def xmem_mm_config(
    mem_every: int = 3,            # write cadence r
    min_mid: int = 5, max_mid: int = 10,  # mid-term buffer [T_min, T_max]
    num_prototypes: int = 128,     # long-term prototypes P
    max_long_term: int = 10000,    # LT_max
    enable_long_term: bool = False,
    deep_update_every: int = 10**9,   # -1 => sync with mem_every
    top_k: int = 30,               # (if used by your repo build)
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
        "hidden_dim": hidden_dim
    }


class XMemMMBackboneWrapper(nn.Module):
    """
    XMem backbone that delegates memory policy to InferenceCore.
    It exposes a per-frame feature vector by hooking the decoder's hidden state.
    """

    def __init__(self,  mm_cfg: Dict, xmem, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.net = xmem

        self.engine = InferenceCore(self.net, config=mm_cfg)   # memory manager
        self.last_hidden = None
        self._hook = None
        self._register_hidden_hook()

    def _register_hidden_hook(self):
        # decoder returns (hidden_state, logits); grab hidden_state on each step
        def _grab_hidden(module, inputs, output):
            self.last_hidden = output[0]  # [B, C, H', W']

        if self._hook is not None:
            self._hook.remove()
        self._hook = self.net.decoder.register_forward_hook(_grab_hidden)

    @torch.no_grad()
    def reset(self):
        # clear memory between sequences (VERY important)
        if hasattr(self.engine, "clear_memory"):
            self.engine.clear_memory()
        self.last_hidden = None

    def forward(self, frames, init_masks=None, init_labels=None, **_):
        B, T, C, H, W = frames.shape
        frames = frames.to(self.device)

        all_feats = []
        for b in range(B):
            self.engine.clear_memory()
            self.last_hidden = None
            self.engine.all_labels = None

            seq_feats = []
            for t in range(T):
                rgb = frames[b, t]                         # [3,H,W]

                if t == 0:
                    if init_masks is not None:
                        # support either [B,K,H,W] or [K,H,W]
                        m = init_masks[b] if (torch.is_tensor(init_masks) and init_masks.ndim == 4) else init_masks
                        m = m.to(self.device).float()      # [K,H,W]
                        # ensure K>=1 and no background channel
                        if m.ndim != 3:
                            raise ValueError(f"init_masks must be [K,H,W], got {m.shape}")
                        lab = init_labels[b] if isinstance(init_labels, list) and isinstance(init_labels[0], list) else init_labels
                        if lab is None:
                            lab = list(range(1, m.size(0)+1))
                    else:
                        # fallback: single full-frame
                        m = torch.ones(1, H, W, device=self.device, dtype=rgb.dtype)
                        lab = [1]
                    self.engine.set_all_labels(lab)
                else:
                    m, lab = None, None

                _ = self.engine.step(rgb, m, lab, end=(t == T - 1))

                if self.last_hidden is None:
                    continue
                hid = self.last_hidden                     # [1,D,H',W']
                feat = hid.view(1, hid.size(1), -1).mean(-1).squeeze(0)
                seq_feats.append(feat)

            if seq_feats:
                all_feats.append(torch.stack(seq_feats, dim=0))
            else:
                D = getattr(self.net, "hidden_dim", 256)
                all_feats.append(torch.zeros(0, D, device=self.device))

        max_Tf = max(f.size(0) for f in all_feats)
        D = all_feats[0].size(-1) if max_Tf > 0 else getattr(self.net, "hidden_dim", 256)
        out = torch.zeros(B, max_Tf, D, device=self.device)
        for b, fb in enumerate(all_feats):
            if fb.numel():
                out[b, :fb.size(0)] = fb
        return out

