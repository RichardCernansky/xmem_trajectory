from torch.utils.data import DataLoader
from .nuscenes_loader import NuScenesLoader
from .collate import collate_varK

class NuScenesDataModule:
    def __init__(self, nusc, train_rows, val_rows, **kwargs):
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.kwargs = kwargs
        self.nusc = nusc

    def train_dataloader(self):
        ds = NuScenesLoader(self.nusc, self.train_rows, **self.kwargs)
        return DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_varK)

    def val_dataloader(self):
        ds = NuScenesLoader(self.nusc, self.val_rows, **self.kwargs)
        return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_varK)
