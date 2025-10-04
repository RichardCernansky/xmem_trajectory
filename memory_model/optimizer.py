import torch

def make_optimizer(MemoryModel) -> torch.optim.Optimizer:
        cfg = MemoryModel.train_config
        lr_head = cfg["lr_head"]
        wd_head = cfg["wd_head"]

        lr_dec = cfg["lr_decoder"]
        wd_dec = cfg["wd_decoder"]

        lr_key = cfg["lr_key_encoder"]
        wd_key = cfg["wd_key_encoder"]

        lr_val = cfg["lr_value_encoder"]
        wd_val = cfg["wd_value_encoder"]

        xmem_core = MemoryModel.xmem_backbone_wrapper.xmem_core
        key_enc = xmem_core.key_encoder
        val_enc = xmem_core.value_encoder
        dec     = xmem_core.decoder

        groups = []
        groups.append({"params": MemoryModel.head.parameters(), "lr": lr_head, "weight_decay": wd_head})

        if dec and any(p.requires_grad for p in dec.parameters()):
            groups.append({"params": [p for p in dec.parameters() if p.requires_grad],
                        "lr": lr_dec, "weight_decay": wd_dec})

        if key_enc and any(p.requires_grad for p in key_enc.parameters()):
            groups.append({"params": [p for p in key_enc.parameters() if p.requires_grad],
                        "lr": lr_key, "weight_decay": wd_key})

        if val_enc and any(p.requires_grad for p in val_enc.parameters()):
            groups.append({"params": [p for p in val_enc.parameters() if p.requires_grad],
                        "lr": lr_val, "weight_decay": wd_val})

        return torch.optim.AdamW(groups, betas=(0.9, 0.999))