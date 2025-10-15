import  torch
import matplotlib.pyplot as plt

from memory_model.model import MemoryModel
from trainer.utils import open_config, open_index
from data.configs.filenames import TRAIN_CONFIG, TRAIN_INDEX, VAL_INDEX, TRAJ_VIS_OUT_PATH
from visualizer.vis_traj import TrajVisualizer

from datamodule.datamodule import NuScenesDataModule
from nuscenes.nuscenes import NuScenes


def run_epoch(model, mode, loader, ep: int):
    
    train_mode = (mode == "train")
    (model.train if train_mode else model.eval)()

    total = 0
    sum_ade = sum_fde = sum_made = sum_mfde = sum_mr = 0.0
    for batch in loader:
        if train_mode:
            m, _ = model.training_step(batch, ep)
        else:
            m, pred_abs_k = model.validation_step(batch)
               # pred_abs_k: (B, K, T, 2) absolute ego XY; mode_probs: (B, K) optional
            viz = TrajVisualizer(save_dir=TRAJ_VIS_OUT_PATH, dpi=150, draw_seams=True)
            #add visualization

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nusc = NuScenes(version="v1.0-trainval", dataroot=r"e:\nuscenes", verbose=False)

    # open training config
    train_config = open_config(TRAIN_CONFIG)
    # load rows from pickle
    train_rows = open_index(TRAIN_INDEX) 
    val_rows = open_index(VAL_INDEX) 

    # Create datasets
    data_module = NuScenesDataModule(nusc, train_rows, val_rows)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = MemoryModel(device)

    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    for ep in range(train_config["epochs"]):
        train_m = run_epoch(model, "train", train_loader, ep)
        val_m   = run_epoch(model, "val", val_loader, ep)
        print(f"Epoch {ep}: Train ADE {train_m['ADE']:.3f} FDE {train_m['FDE']:.3f} | "
              f"Val ADE {val_m['ADE']:.3f} FDE {val_m['FDE']:.3f} MR@2m {val_m['MR@2m']:.3f}")

        hist["epoch"].append(ep)
        hist["train_ADE"].append(train_m["ADE"])
        hist["train_FDE"].append(train_m["FDE"])
        hist["val_ADE"].append(val_m["ADE"])
        hist["val_FDE"].append(val_m["FDE"])
        hist["val_MR2"].append(val_m["MR@2m"])

    plt.figure(figsize=(8,5))
    plt.plot(hist["epoch"], hist["train_ADE"], marker="o", label="Train ADE")
    plt.plot(hist["epoch"], hist["val_ADE"], marker="o", label="Val ADE")
    plt.plot(hist["epoch"], hist["train_FDE"], marker="o", label="Train FDE")
    plt.plot(hist["epoch"], hist["val_FDE"], marker="o", label="Val FDE")
    plt.xlabel("Epoch"); plt.ylabel("Error"); plt.title("ADE/FDE"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("data/runs/ade_fde.png")

    plt.figure(figsize=(8,4))
    plt.plot(hist["epoch"], hist["val_MR2"], marker="o", label="Val MR@2m")
    plt.xlabel("Epoch"); plt.ylabel("Miss rate"); plt.title("Miss Rate @ 2 m"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("data/runs/missrate.png")


if __name__ == "__main__":
    main()
