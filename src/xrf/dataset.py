import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import gain_neighborhood_band


class SpectraDataset(Dataset):
    def __init__(
        self,
        npz_path,
        mode=None,
        peak_prediction=False,
        near_band=None,
    ):
        data = np.load(npz_path, mmap_mode="r")
        self.mode = mode
        self.peak_prediction = peak_prediction
        self.spectra = data["spectra"].astype(np.float32)
        self.endmembers = data["endmembers"].astype(np.float32)
        if near_band is not None:
            self.spectra = gain_neighborhood_band(self.spectra, near_band)
        self.energy = data["energy"].astype(np.float32)
        if mode == "identification":
            self.components = data["components"].astype(np.float32)
            self.labels = (data["components"] > 0).astype(np.float32)
        elif mode == "unmixing":
            self.components = data["components"].astype(np.float32)
        elif self.peak_prediction:
            self.num_peaks = data["num_peaks"].astype(np.float32)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = torch.from_numpy(self.spectra[idx])
        if self.mode == "identification":
            labels = torch.from_numpy(self.labels[idx])
            components = torch.from_numpy(self.components[idx])
            return spectrum, labels
        elif self.mode == "unmixing":
            components = torch.from_numpy(self.components[idx])
            return spectrum, components
        elif self.peak_prediction:
            return spectrum, torch.from_numpy(self.num_peaks[idx])
        return spectrum
