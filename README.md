# XMem Trajectory Extension

## Folder Structure

```plaintext
XMem/
 ├─ xmem/                
 │    └─ ...              # Original XMem implementation
 ├─ traj/                 # Trajectory prediction extension
 │    ├─ predictor.py     # Wrapper that uses XMem encoders + memory
 │    ├─ head.py          # Small GRU/MLP head
 │    └─ datamodules.py   # Lightweight nuScenes sequence loader (stub)
 └─ train_traj.py         # Training script



XMEM PIPELINE #1 - PSEUDO
assumptions:
    - no long term memory yet, xmem selection mechanism
    - fixed size of future timestamps
    - using ego vehicle coordinates
    - mask computation on-the-fly

    - paralelize the training loops 
    - resonnet into 2d and into memory and turn on long term
    - decide on reimplementation of prioritizing prototypes
    - improve classification head 


*Prefatched Data index row: {
                    "scene_name": scene_name,
                    "start_sample_token": obs_tokens[0],
                    "obs_sample_tokens": obs_tokens,
                    "fut_sample_tokens": fut_tokens,
                    "cam": cam,
                    "cam_sd_tokens": cam_sd_tokens,   # len = t_in
                    "img_paths": img_paths,           # len = t_in
                    "intrinsics": intrinsics,         # len = t_in, 3x3 each

                    "target": {
                        "agent_id": inst_tok,
                        "last_xy": last_xy_e,         # EGO frame at t_in-1
                        "future_xy": fut_xy_e,        # EGO frame, len = t_out
                        "frame": "ego_xy"
                    },

                    "context": {
                        "t0_cam_sd_token": cam_sd_tokens[0],  # for on-the-fly masks
                        "anchor_sd_token": sd_anchor          # ego frame anchor (optional)
                    }
                }

*def xmem_backbone.step(frames, init_masks, init_labels):
    B, T, C, H, W = frames.shape # Batch, Timestamp, Channel, Height, Width
    all_features = []
    for b in range(B):
        memory.clear_memory()
        mask0 = init_masks[0]
        label0 = init_labels[0]
        sequence_features = []

        for t in range(T):
            rgb = frames[b, t]
            if t == 0:
                memory.set_labels(label0)
                memory.step(rgb, m0, lab0, end=(t == T - 1)) #implicitly hook hidden features
            else:
                memory.step(rgb, None, None, end=(t == T - 1)) #implicitly hook hidden features

            hid = self.last_hidden  # shape: [1, K, C, H', W']
            features = hid.mean(dim=(1, 3, 4)).squeeze(0)  # -> [C]
            sequence_features.append(features)
        
        all_features.append(torch.stack(sequence_features))

    out = torch_zeros.allocate([B,max_Tf,C])
    for i, b in enumerate(all_features):
        out[i] = b
    return out   # [B,max_Tf,C] batch of features


*Training:
    for n epochs:
        for batch in train_loader:
            features = xmem_backbone.step(frames, init_masks, init_labels)
            pred = head(features) # -> GRU -> Linear -> preds
            ade, fde, loss = ade_fde_loss(pred_abs, gt_future)
            optimizer.step()
        
        print(loss)


*Output example:
    Epoch 0: ADE 0.143  FDE 0.277