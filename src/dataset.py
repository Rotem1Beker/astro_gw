# PyTorch Dataset
import torch
from torch.utils.data import Dataset
from simulate import simulate
from preprocess import preprocess

class GWDataset(Dataset):
    def __init__(self, n_samples, config):
        self.data = []
        for _ in range(n_samples):
            x, theta = simulate(
                config["data"]["sample_rate"],
                config["data"]["duration"],
                config["data"]["noise_std"],
            )
            X = preprocess(x)
            self.data.append((X, theta))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, theta = self.data[idx]
        return torch.tensor(X), torch.tensor(theta, dtype=torch.float32)
