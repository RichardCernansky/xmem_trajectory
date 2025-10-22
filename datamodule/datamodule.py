from torch.utils.data import DataLoader
from .nuscenes_loader import NuScenesLoader
from .collate import collate_varK
from torch.utils.data import DataLoader
import os

class NuScenesDataModule:
    def __init__(self, nusc, train_rows, val_rows, **kwargs):
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.kwargs = kwargs
        self.nusc = nusc
        # sensible Windows defaults; tune if your CPU/disk can handle more
        self.num_workers = max(2, os.cpu_count() // 2)
        self.pin_memory = True
        self.prefetch_factor = 2          # needs num_workers > 0
        self.persistent_workers = True    # keeps workers warm across epochs

    def train_dataloader(self):
        ds = NuScenesLoader(self.nusc, self.train_rows, **self.kwargs)
        return DataLoader(
            ds,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_varK,      # keep light; do heavy work in __getitem__
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            drop_last=True,               # avoids tiny last batch stalls
        )

    def val_dataloader(self):
        ds = NuScenesLoader(self.nusc, self.val_rows, **self.kwargs)
        return DataLoader(
            ds,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_varK,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )



