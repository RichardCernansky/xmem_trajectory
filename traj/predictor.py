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
    single_object: bool = False
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
        "hidden_dim": 256
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

    @torch.no_grad()
    def forward(
        self,
        frames: torch.Tensor,                          # [B,T,C,H,W]
        masks: Optional[List[Optional[torch.Tensor]]] = None,
        labels: Optional[List] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        frames = frames.to(self.device)

        all_feats = []

        for b in range(B):
            # reset per sequence
            self.engine.clear_memory()
            self.last_hidden = None
            self.engine.all_labels = None

            seq_feats = []
            for t in range(T):
                # XMem expects unbatched image [C,H,W]
                rgb = frames[b, t]  # [C,H,W]

                # mask for this (b,t)
                if t == 0:
                    # foreground everywhere (you can replace with your real object mask)
                    fg = torch.ones(H, W, device=self.device, dtype=frames.dtype)
                    bg = 1.0 - fg
                    m = torch.stack([bg, fg], dim=0)   # [2, H, W]  -> matches 5-channel input (3+2)
                    lab = [1]                          # label for the foreground object
                    self.engine.set_all_labels(lab)
                else:
                    m = None
                    lab = None


                # run one step (end=True on last frame)
                # _ = self.engine.step(rgb, m, lab, end=(t == T - 1))

                                # run one step; end=True on last frame
                try:

                    _ = self.engine.step(rgb, m, lab, end=(t == T - 1))
                except Exception as e:
                    # helpful debug info if anything still goes wrong
                    print(
                        f"[XMem step ERROR] b={b} t={t} "
                        f"rgb={tuple(rgb.shape)} "
                        f"mask={(None if m is None else tuple(m.shape))} "
                        f"lab={lab} all_labels={self.engine.all_labels}"
                    )
                    raise

                # collect hidden feature captured by hook
                if self.last_hidden is None:
                    continue
                hid = self.last_hidden           # [1, D, H’, W’] (engine adds batch=1 internally)
                feat = hid.view(1, hid.size(1), -1).mean(-1).squeeze(0)  # [D]
                seq_feats.append(feat)

            if len(seq_feats) == 0:
                D = getattr(self.net, "hidden_dim", 256)
                all_feats.append(torch.zeros(0, D, device=self.device))
            else:
                all_feats.append(torch.stack(seq_feats, dim=0))  # [T_feat,D]

        # pad to common T_feat (optional)
        max_Tf = max(f.size(0) for f in all_feats)
        D = all_feats[0].size(-1) if max_Tf > 0 else getattr(self.net, "hidden_dim", 256)
        out = torch.zeros(B, max_Tf, D, device=self.device)
        for b, fb in enumerate(all_feats):
            if fb.numel():
                out[b, :fb.size(0)] = fb
        return out  # [B,T_feat,D]
