Folder structure:
XMem/
 ├─ xmem/                # original
 ├─ traj/                # <-- new
 │   ├─ predictor.py     # wrapper that uses XMem encoders+memory
 │   ├─ head.py          # small GRU/MLP head
 │   └─ datamodules.py   # lightweight nuScenes sequence loader (stub)
 └─ train_traj.py        # your training script

# xmem_trajectory


XMEM PIPELINE #1 - PSEUDO

*Prefatched Data index: {
        "frames": frames,
        "traj": traj,
        "last_pos": last,
        "init_masks": init_masks,
        "init_labels": init_labels,
        "meta": meta,
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