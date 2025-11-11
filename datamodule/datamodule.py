# from torch.utils.data import DataLoader
# from .nuscenes_loader import NuScenesLoader
# from .collate import collate_varK
# import os, torch

# class NuScenesDataModule:
#     def __init__(
#         self,
#         nusc,
#         train_rows,
#         val_rows,
#         batch_size: int = 4,
#         num_workers: int = 4,
#         prefetch_factor: int = 4,
#         pin_memory: bool = torch.cuda.is_available(),
#         persistent_workers: bool = True,
#         **kwargs,
#     ):
#         self.train_rows = train_rows
#         self.val_rows = val_rows
#         self.kwargs = kwargs
#         self.nusc = nusc

#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.prefetch_factor = prefetch_factor
#         self.pin_memory = pin_memory
#         self.persistent_workers = persistent_workers and num_workers > 0

#     def _loader(self, ds, shuffle: bool):
#         return DataLoader(
#             ds,
#             batch_size=self.batch_size,
#             shuffle=shuffle,
#             collate_fn=collate_varK,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             persistent_workers=self.persistent_workers,
#             prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
#         )

#     def train_dataloader(self):
#         ds = NuScenesLoader(self.nusc, self.train_rows, **self.kwargs)
#         return self._loader(ds, shuffle=True)

#     def val_dataloader(self):
#         ds = NuScenesLoader(self.nusc, self.val_rows, **self.kwargs)
#         return self._loader(ds, shuffle=False)



from torch.utils.data import DataLoader
from .nuscenes_loader import NuScenesLoader
from .collate import collate_varK

class NuScenesDataModule:
    def __init__(self, nusc, train_rows, val_rows, batch_size, **kwargs):
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.kwargs = kwargs
        self.nusc = nusc
        self.batch_size = batch_size

    def train_dataloader(self):
        ds = NuScenesLoader(self.nusc, self.train_rows, **self.kwargs)
        return DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_varK)

    def val_dataloader(self):
        ds = NuScenesLoader(self.nusc, self.val_rows, **self.kwargs)
        return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_varK)
