import torch

def make_optimizer(model) -> torch.optim.Optimizer:
    cfg = model.train_config
    def get(k, default): return cfg[k] if k in cfg else default

    groups = []
    added = set()

    def add_group(params, lr, wd):
        ps = [p for p in params if p.requires_grad and id(p) not in added]
        if not ps:
            return
        groups.append({"params": ps, "lr": float(lr), "weight_decay": float(wd)})
        for p in ps: added.add(id(p))

    # --- Head ---
    add_group(model.head.parameters(), get("lr_head", 1e-3), get("wd_head", 1e-4))

    # --- Depth path (late concat) ---
    if hasattr(model, "depth_encoder"):
        add_group(model.depth_encoder.parameters(),
                  get("lr_depth", get("lr_head", 1e-3)),
                  get("wd_depth", get("wd_head", 1e-4)))
    if hasattr(model, "fuser"):
        add_group(model.fuser.parameters(),
                  get("lr_fuser", get("lr_depth", get("lr_head", 1e-3))),
                  get("wd_fuser", get("wd_depth", get("wd_head", 1e-4))))

    # --- XMem parts (only if unfrozen) ---
    xw = getattr(model, "xmem_backbone_wrapper", None)
    xcore = getattr(xw, "xmem_core", None)

    if xcore is not None:
        key_enc = getattr(xcore, "key_encoder", None)
        val_enc = getattr(xcore, "value_encoder", None)
        dec     = getattr(xcore, "decoder", None)

        if key_enc is not None:
            add_group(key_enc.parameters(), get("lr_key_encoder", 1e-5), get("wd_key_encoder", 1e-4))
        if val_enc is not None:
            add_group(val_enc.parameters(), get("lr_value_encoder", 1e-5), get("wd_value_encoder", 1e-4))
        if dec is not None:
            add_group(dec.parameters(), get("lr_decoder", 1e-5), get("wd_decoder", 1e-4))

        # any other unfrozen params inside the wrapper (e.g., you unfreeze the last stage)
        other = [p for _, p in xw.named_parameters() if p.requires_grad and id(p) not in added]
        add_group(other, get("lr_xmem_other", get("lr_decoder", 1e-5)),
                        get("wd_xmem_other", get("wd_decoder", 1e-4)))

    # --- Safety net: leftover trainable params (unlikely, but future-proof) ---
    leftovers = [p for p in model.parameters() if p.requires_grad and id(p) not in added]
    add_group(leftovers, get("lr_misc", get("lr_head", 1e-3)), get("wd_misc", get("wd_head", 1e-4)))

    return torch.optim.AdamW(groups, betas=(0.9, 0.999))
