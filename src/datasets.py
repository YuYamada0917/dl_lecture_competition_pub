import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy import signal
from sklearn.preprocessing import StandardScaler

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", preprocess: bool = True, 
                 original_rate: int = 1000, target_rate: int = 250, 
                 low: float = 1.0, high: float = 40.0) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        if preprocess:
            self.X = self.preprocess_data(self.X, original_rate, target_rate, low, high)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def preprocess_data(self, X, original_rate, target_rate, low, high):
        X_np = X.numpy()

        # 1. Resampling
        X_resampled = signal.resample(X_np, int(X_np.shape[2] * target_rate / original_rate), axis=2)

        # 2. Filtering
        nyquist = 0.5 * target_rate
        low_normalized = low / nyquist
        high_normalized = min(high, nyquist-1) / nyquist  # Ensure high frequency is below Nyquist
        if low_normalized < high_normalized:
            b, a = signal.butter(4, [low_normalized, high_normalized], btype='band')
            X_filtered = signal.filtfilt(b, a, X_resampled, axis=2)
        else:
            print(f"Warning: Invalid filter frequencies. Using unfiltered data.")
            X_filtered = X_resampled

        # 3. Baseline correction
        baseline_samples = int(0.1 * target_rate)
        baseline = X_filtered[:, :, :baseline_samples].mean(axis=2, keepdims=True)
        X_baseline_corrected = X_filtered - baseline

        # 4. Scaling
        scaler = StandardScaler()
        X_scaled = np.zeros_like(X_baseline_corrected)
        for i in range(X_baseline_corrected.shape[0]):
            X_scaled[i] = scaler.fit_transform(X_baseline_corrected[i].T).T

        X_preprocessed = torch.from_numpy(X_scaled).float()

        return X_preprocessed