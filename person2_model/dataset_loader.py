import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):

    def __init__(self, folder):

        self.files = [
            f for f in os.listdir(folder)
            if f.endswith(".npy")
        ]

        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        path = os.path.join(self.folder, filename)

        sequence = np.load(path)

        label = filename.split("_")[0]

        return (
            torch.tensor(sequence, dtype=torch.float32),
            label
        )