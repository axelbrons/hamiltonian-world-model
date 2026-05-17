import torch
from torch.utils.data import Dataset

class GymDataset(Dataset):
    def __init__(self, data_array):
        # data_array est ton tableau numpy contenant tes milliers d'images 64x64
        self.data = torch.tensor(data_array, dtype=torch.float32)
        
        # Normaliser les pixels de [0, 255] vers [0, 1]
        self.data = self.data / 255.0 
        
        # PyTorch veut le format (Channels, Height, Width), donc (3, 64, 64)
        # S'assurer que les données sont dans ce format
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] # Le VAE ne prend que l'image, pas de "label"