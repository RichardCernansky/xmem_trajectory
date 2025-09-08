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
    
    def _pool_hidden(self, hid: torch.Tensor) -> torch.Tensor:
        if hid.dim() == 5:
            return hid.mean(dim=(1, 3, 4)).squeeze(0)
        if hid.dim() == 4:
            return hid.mean(dim=(2, 3)).squeeze(0)
        raise RuntimeError(f"Unexpected hidden shape: {tuple(hid.shape)}")

    @torch.no_grad()
    def reset(self):
        # clear memory between sequences (VERY important)
        if hasattr(self.engine, "clear_memory"):
            self.engine.clear_memory()
        self.last_hidden = None

    def forward(                           # inputs as batched data
        self,
        frames: torch.Tensor,              # [B,T,3,H,W] batch of RGB frame sequences
        init_masks: List[torch.Tensor],    # list of [K_i,H,W] masks for t=0, per sample
        init_labels: List[List[int]],      # list of lists of labels aligned with masks
        **_,                               # ignore extra keys from collate (traj, meta, etc.)
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape       # unpack sequence dimensions
        frames = frames.to(self.device)    # move frames to target device (GPU)

        all_feats = []                     # will collect per-sequence features
        for b in range(B):                 # loop over batch dimension
            self.engine.clear_memory()     # reset XMem memory for new sequence
            self.last_hidden = None        # clear hidden state hook
            self.engine.all_labels = None  # clear label assignment

            seq_feats = []                 # will collect features across time for this sequence
            m0 = init_masks[b].to(self.device).float()  # t=0 masks [K_i,H,W] on device
            lab0 = [int(x) for x in init_labels[b]]     # convert labels to plain ints

            for t in range(T):             # loop over time steps
                rgb = frames[b, t]         # current frame [3,H,W]

                if t == 0:                 # first frame: provide initialization masks+labels
                    self.engine.set_all_labels(lab0)                # register object IDs
                    _ = self.engine.step(rgb, m0, lab0, end=(t == T - 1))
                else:                      # later frames: no masks, tracker propagates
                    _ = self.engine.step(rgb, None, None, end=(t == T - 1))

                if self.last_hidden is not None:        # hook captured decoder hidden state
                    hid = self.last_hidden              # [K,C,H',W']
                    feat = self._pool_hidden(hid)
                    # → global pooled feature vector [C]
                    seq_feats.append(feat)

            if seq_feats:                               # stack features for this sequence
                all_feats.append(torch.stack(seq_feats, dim=0))  # [T_f,C]
            else:                                       # edge case: no features
                D = getattr(self.net, "hidden_dim", 256)
                all_feats.append(torch.zeros(0, D, device=self.device))

        max_Tf = max(f.size(0) for f in all_feats)      # maximum time length across batch
        D = all_feats[0].size(-1) if max_Tf > 0 else getattr(self.net, "hidden_dim", 256)
        out = torch.zeros(B, max_Tf, D, device=self.device)  # allocate [B,max_Tf,C]

        for b, fb in enumerate(all_feats):              # copy each seq into padded tensor
            if fb.numel():
                out[b, :fb.size(0)] = fb
        return out                                      # [B,max_Tf,C] batch of features
