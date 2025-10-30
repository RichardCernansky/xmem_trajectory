import argparse
import os
import torch
import matplotlib.pyplot as plt

from memory_model.model import MemoryModel
from trainer.utils import open_config, open_index
from data.configs.filenames import TRAIN_CONFIG, TRAJ_VIS_OUT_PATH
from visualizer.vis_traj import TrajVisualizer
from datamodule.datamodule import NuScenesDataModule
from nuscenes.nuscenes import NuScenes

def parse_args():
    p = argparse.ArgumentParser()
    # dataset + indices
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--dataroot", required=True, help="NuScenes root directory")
    p.add_argument("--train_index", required=True, help="Path to train index .pkl")
    p.add_argument("--val_index",   required=True, help="Path to val index .pkl")
    # outputs
    p.add_argument("--checkpoints_dir", required=True, help="Directory to save checkpoints")
    # run options
    p.add_argument("--model_name", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


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
            viz = TrajVisualizer(save_dir=TRAJ_VIS_OUT_PATH, dpi=150, draw_seams=True)
            # Optional: viz.plot(pred_abs_k, batch["traj"], save=True)

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
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = args.checkpoints_dir
    os.makedirs(ckpt_dir, exist_ok=True)              # create if missing
    ckpt_path = os.path.join(ckpt_dir, f"{args.model_name}.pth")

    # NuScenes
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)

    # Load config & indices
    train_config = open_config(TRAIN_CONFIG)
    train_rows = open_index(args.train_index)
    val_rows   = open_index(args.val_index)

    data_module = NuScenesDataModule(nusc, train_rows, val_rows)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = MemoryModel(device)
    start_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("Starting new training run.")

    hist = {"epoch": [], "train_ADE": [], "train_FDE": [], "val_ADE": [], "val_FDE": [], "val_MR2": []}

    for ep in range(start_epoch, train_config["epochs"]):
        print(f"--- Epoch {ep} ---")
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

        # ✅ Save model every epoch
        torch.save({
            "epoch": ep,
            "model_state_dict": model.state_dict(),
        }, ckpt_path)
        print(f"Saved model to {ckpt_path}")

    # Plot metrics
    plt.figure(figsize=(8,5))
    plt.plot(hist["epoch"], hist["train_ADE"], marker="o", label="Train ADE")
    plt.plot(hist["epoch"], hist["val_ADE"], marker="o", label="Val ADE")
    plt.plot(hist["epoch"], hist["train_FDE"], marker="o", label="Train FDE")
    plt.plot(hist["epoch"], hist["val_FDE"], marker="o", label="Val FDE")
    plt.xlabel("Epoch"); plt.ylabel("Error"); plt.title("ADE/FDE"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("data/runs/ade_fde.png")

    plt.figure(figsize=(8,4))
    plt.plot(hist["epoch"], hist["val_MR2"], marker="o", label="Val MR@2m")
    plt.xlabel("Epoch"); plt.ylabel("Miss rate"); plt.title("Miss Rate @ 2 m")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("data/runs/missrate.png")


if __name__ == "__main__":
    main()
