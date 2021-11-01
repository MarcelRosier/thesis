import os

import numpy as np
import torch
from torch._C import dtype
from constants import (ENV, SYN_TUMOR_BASE_PATH,
                       SYN_TUMOR_PATH_TEMPLATE, TUMOR_SUBSET_1K, TUMOR_SUBSET_200,)
from torch.utils.data import Dataset

SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]


class TumorT1CDataset(Dataset):
    """Tumor T1C dataset."""

    def __init__(self, subset=None, transform=None):
        """TODO"""
        folders = os.listdir(SYN_TUMOR_BASE_PATH)
        folders.sort(key=lambda f: int(f))
        if subset:
            folders = folders[subset[0]: subset[1]]
        else:
            folders = folders[:TUMOR_SUBSET_200]
        self.n_samples = len(folders)
        self.tumor_ids = folders
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        tumor = self.load_single_tumor(tumor_id=self.tumor_ids[idx])
        tensor = torch.from_numpy(tumor)
        # cast double to float to match weight dtype
        tensor = tensor.float()
        # add channel dim
        tensor.unsqueeze_(0)
        return tensor, torch.empty(0)

    def load_single_tumor(self, tumor_id):
        tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(id=tumor_id))[
            'data']

        # crop 129^3 to 128^3 if needed
        if tumor.shape != (128, 128, 128):
            tumor = np.delete(np.delete(
                np.delete(tumor, 128, 0), 128, 1), 128, 2)

        # normalize
        max_val = tumor.max()
        if max_val != 0:
            tumor *= 1.0/max_val

        # threshold
        tumor[tumor < 0.6] = 0
        tumor[tumor >= 0.6] = 1
        return tumor
