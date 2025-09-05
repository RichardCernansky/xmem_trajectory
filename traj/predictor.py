import torch
import torch.nn as nn

class XMemBackboneWrapper(nn.Module):
    def __init__(self, xmem_model, write_interval=3):
        super().__init__()
        self.xmem = xmem_model
        self.write_interval = write_interval

        # Memory banks (dynamic, not learnable params)
        self.memory_key = None
        self.memory_shrinkage = None
        self.memory_value = None

    @torch.no_grad()
    def reset_memory(self):
        """Clear memory between sequences."""
        self.memory_key = None
        self.memory_shrinkage = None
        self.memory_value = None

    @torch.no_grad()
    def forward(self, frames, masks=None):
        """
        frames: list of [B,3,H,W] tensors (T_in long)
        masks:  list of [B,1,H,W] tensors for first frame (optional)
        Returns: [B, T_in, D] pooled features per timestep
        """
        B = frames[0].size(0)
        per_t_feats = []
        hidden_state = None

        for t, frame in enumerate(frames):
            # 1) Encode current frame to get key + multiscale features
            key, shrinkage, selection, f16, f8, f4 = self.xmem.encode_key(frame)

            # 2) For first frame, also encode value (needs mask) TODO: implement masks with bounding boxes form Nuscenes
            if t == 0:
                B, _, H, W = frame.shape
                dummy_mask = torch.ones(B, 1, H, W, device=frame.device, dtype=frame.dtype)
                g16, h16 = self.xmem.encode_value(frame, f16, hidden_state, dummy_mask)
                self.memory_key = key.unsqueeze(2)          # add T dim
                self.memory_shrinkage = shrinkage.unsqueeze(2) if shrinkage is not None else None
                self.memory_value = g16.unsqueeze(2)        # add T dim
                hidden_state = h16
                continue

            # 3) Read from memory if it exists
            if self.memory_key is not None:
                memory_readout = self.xmem.read_memory(
                    query_key=key,
                    query_selection=selection,
                    memory_key=self.memory_key,
                    memory_shrinkage=self.memory_shrinkage,
                    memory_value=self.memory_value
                )
            else:
                memory_readout = torch.zeros(
                    (B, 1, self.xmem.value_dim, *f16.shape[-2:]),
                    device=frame.device, dtype=frame.dtype
                )

            # 4) Run decoder to get hidden features (ignore mask logits)
            hidden_state, logits, prob = self.xmem.segment(
                multi_scale_features=(f16, f8, f4),
                memory_readout=memory_readout,
                hidden_state=hidden_state,
                selector=None,
                h_out=True,
                strip_bg=True
            )

            # logits: [B,1,H,W], prob: [B,1,H,W], hidden_state: [B,C,H,W]
            # Use hidden_state as the memory-augmented feature map
            feat_vec = hidden_state.flatten(2).mean(dim=-1)   # [B, D=C] where D is feature dimensionality = previous channel dimension
            per_t_feats.append(feat_vec)
            

            # 5) Optionally write this frame into memory
            if (t % self.write_interval) == 0:
                g16, h16 = self.xmem.encode_value(frame, f16, hidden_state, masks=None)
                self.memory_key = torch.cat([self.memory_key, key.unsqueeze(2)], dim=2)
                self.memory_value = torch.cat([self.memory_value, g16.unsqueeze(2)], dim=2)
                if shrinkage is not None:
                    self.memory_shrinkage = torch.cat([self.memory_shrinkage, shrinkage.unsqueeze(2)], dim=2)

        # a sequence of per-frame feature vectors ready for your trajectory head (GRU/MLP → ADE/FDE).
        return torch.stack(per_t_feats, dim=1)  # [B, T_in, D]
    
