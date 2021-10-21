import os

import numpy as np
import torch
from torch.functional import Tensor
from constants import (IS_LOCAL, SYN_TUMOR_BASE_PATH_LOCAL,
                       SYN_TUMOR_PATH_TEMPLATE_SERVER, TUMOR_SUBSET_1K, TUMOR_SUBSET_200, SYN_TUMOR_PATH_TEMPLATE_SERVER, SYN_TUMOR_PATH_TEMPLATE_LOCAL)
from torch.utils.data import Dataset

SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH_LOCAL if IS_LOCAL else SYN_TUMOR_PATH_TEMPLATE_SERVER
SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE_LOCAL if IS_LOCAL else SYN_TUMOR_PATH_TEMPLATE_SERVER


class TumorT1CDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """TODO"""
        folders = os.listdir(SYN_TUMOR_BASE_PATH)
        folders = folders[:TUMOR_SUBSET_1K]
        self.n_samples = len(folders)
        folders.sort(key=lambda f: int(f))
        self.tumor_ids = folders
        # data = np.empty([self.n_samples, 128, 128, 128])
        # for i, tumor_id in enumerate(folders):
        #     data[i] = self.load_single_tumor(tumor_id=tumor_id)
        # self.dataset = torch.from_numpy(data)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        tumor = self.load_single_tumor(tumor_id=self.tumor_ids[idx])
        return torch.from_numpy(tumor), torch.empty(0)

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
