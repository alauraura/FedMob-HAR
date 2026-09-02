import torch
import numpy as np
from torch.utils.data import Dataset

def apply_gaussian_noise(X, mean=0.0, std=0.5):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def apply_scaling(X, scale_factor=1.5):
    return X * scale_factor

def apply_sensor_drop(X, start_col, end_col):
    X_dropped = X.copy()
    # Corrige o fatiamento de 3D para 2D (timesteps, canais)
    X_dropped[:, start_col:end_col] = 0.0
    return X_dropped

class HARDatasetFederado(Dataset):
    def __init__(self, X, y, contexto="base"):
        self.X = X
        self.y = y
        self.contexto = contexto

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        janela = self.X[idx].copy()
        label = self.y[idx]

        if self.contexto == "ruido":
            janela = apply_gaussian_noise(janela, std=0.8)
        elif self.contexto == "escala":
            janela = apply_scaling(janela, scale_factor=0.3)
        elif self.contexto == "falha_sensor":
            janela = apply_sensor_drop(janela, start_col=0, end_col=13)

        tensor_x = torch.tensor(janela, dtype=torch.float32)
        tensor_y = torch.tensor(label, dtype=torch.long)
        
        return tensor_x, tensor_y