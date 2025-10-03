import os, sys, torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle
import json

from memory_model.model import MemoryModel

#put whre the plots are created
os.environ["MPLBACKEND"] = "Agg"   # optional but safe
import matplotlib
matplotlib.use("Agg")              # must be before pyplot
import matplotlib.pyplot as plt


#old
from datamodule.datamodules import NuScenesSeqLoader, collate_varK
from nuscenes.nuscenes import NuScenes


def run_epoch(model, mode, loader):
    
    train_mode = (mode == "train")
    (model.train if train_mode else model.eval)()

    total = 0
    sum_ade = sum_fde = sum_made = sum_mfde = sum_mr = 0.0
    for batch in loader:
        if train_mode:
            m, _ = model.training_step(batch)
        else:
            m, _ = model.validation_step(batch)

        bsz = batch["traj"].shape[0]
        sum_ade  += m["ADE"]   * bsz
        sum_fde  += m["FDE"]   * bsz
        sum_made += m["mADE"]  * bsz
        sum_mfde += m["mFDE"]  * bsz
        sum_mr   += m["MR@2m"] * bsz
        total    += bsz


    return {
        "ADE":  sum_ade  / max(1, total),
        "FDE":  sum_fde  / max(1, total),
        "mADE": sum_made / max(1, total),
        "mFDE": sum_mfde / max(1, total),
        "MR@2m": sum_mr  / max(1, total),
    }



def main():
    train_config = json.loads("data/configs/train_config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=False)

    # load rows from pickle
    with open("train_agents_index.pkl", "rb") as f:
        train_rows = pickle.load(f)
    with open("val_agents_index.pkl", "rb") as f:
        val_rows = pickle.load(f)

    # Create datasets
    train_ds = NuScenesSeqLoader(nusc=nusc, rows=train_rows, out_size=(384, 640))
    val_ds   = NuScenesSeqLoader(nusc=nusc, rows=val_rows,   out_size=(384, 640))
    train_loader = DataLoader(train_ds, batch_size=train_config["batch_size"], shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_varK)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_varK)

    model = MemoryModel(device)

    
    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    for ep in range(train_config["epochs"]):
        train_m = run_epoch(model, "train", train_loader)
        val_m   = run_epoch(model, "val", val_loader)
        print(f"Epoch {ep}: Train ADE {train_m['ADE']:.3f} FDE {train_m['FDE']:.3f} | "
              f"Val ADE {val_m['ADE']:.3f} FDE {val_m['FDE']:.3f} MR@2m {val_m['MR@2m']:.3f}")

        hist["epoch"].append(ep)
        hist["train_ADE"].append(train_m["ADE"])
        hist["train_FDE"].append(train_m["FDE"])
        hist["val_ADE"].append(val_m["ADE"])
        hist["val_FDE"].append(val_m["FDE"])
        hist["val_MR2"].append(val_m["MR@2m"])

        with torch.no_grad():
            try:
                batch = next(iter(val_loader))
                frames     = batch["frames"].to(device, non_blocking=True)
                init_masks = batch["init_masks"]
                init_labels= batch["init_labels"]
                gt_future  = batch["traj"].to(device, non_blocking=True)
                last_pos   = batch["last_pos"].to(device, non_blocking=True)

                feats = backbone(frames, init_masks=init_masks, init_labels=init_labels)
                traj_res_k, mode_logits = head(feats)                  # [B,K,T,2]

                pred_abs_k = last_pos[:, None, None, :] + traj_res_k   # absolute coords

                visualize_predictions(
                    pred_abs_k, gt_future, last_pos,
                    save_path=f"runs/vis_epoch_{ep}.png", max_samples=8
                )
            except StopIteration:
                pass

    plt.figure(figsize=(8,5))
    plt.plot(hist["epoch"], hist["train_ADE"], marker="o", label="Train ADE")
    plt.plot(hist["epoch"], hist["val_ADE"], marker="o", label="Val ADE")
    plt.plot(hist["epoch"], hist["train_FDE"], marker="o", label="Train FDE")
    plt.plot(hist["epoch"], hist["val_FDE"], marker="o", label="Val FDE")
    plt.xlabel("Epoch"); plt.ylabel("Error"); plt.title("ADE/FDE"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("runs/ade_fde.png")

    plt.figure(figsize=(8,4))
    plt.plot(hist["epoch"], hist["val_MR2"], marker="o", label="Val MR@2m")
    plt.xlabel("Epoch"); plt.ylabel("Miss rate"); plt.title("Miss Rate @ 2 m"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("runs/missrate.png")


if __name__ == "__main__":
    main()
