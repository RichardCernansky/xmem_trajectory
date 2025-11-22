import os
import torch

from trainer.utils import open_config, open_index
from data.configs.filenames import TRAIN_PREDICTOR_CONFIG
from datamodule.datamodule import NuScenesDataModule, load_nuscenes
from my_model.model import MemoryModel


def run_epoch_mask(model, loader, epoch: int, train: bool):
    if train:
        model.train()
        grad_ctx = torch.enable_grad()
    else:
        model.eval()
        grad_ctx = torch.inference_mode()

    metrics_sum = {}
    n_batches = 0

    with grad_ctx:
        for batch in loader:
            if train:
                m, _ = model.training_step(batch, epoch)
            else:
                m, _ = model.validation_step(batch)

            n_batches += 1
            for k, v in m.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)

    metrics_avg = {k: v / n_batches for k, v in metrics_sum.items()}
    return metrics_avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = open_config(TRAIN_PREDICTOR_CONFIG)

    ckpt_path = train_cfg["xmem_model"]
    ckpt_dir = os.path.dirname(ckpt_path) if ckpt_path else "."
    if ckpt_dir != "":
        os.makedirs(ckpt_dir, exist_ok=True)
    vis_path = ckpt_dir

    nusc = load_nuscenes(
        dataroot=train_cfg["nusc_dataroot"],
        version=train_cfg["nusc_version"],
    )

    train_rows = open_index(train_cfg["train_index"])
    val_rows = open_index(train_cfg["val_index"])

    data_module = NuScenesDataModule(
        nusc,
        train_rows,
        val_rows,
        batch_size=train_cfg["batch_size"],
    )
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = MemoryModel(device=device, vis_path=vis_path, train_config=train_cfg)
    model.first_stage = True

    num_epochs = int(train_cfg.get("xmem_epochs", 10))

    for ep in range(num_epochs):
        train_metrics = run_epoch_mask(model, train_loader, ep, train=True)
        val_metrics = run_epoch_mask(model, val_loader, ep, train=False)

        torch.save(model.xmem.xmem_core.state_dict(), ckpt_path)
        print(f"epoch {ep}: train={train_metrics}, val={val_metrics}")


if __name__ == "__main__":
    main()
