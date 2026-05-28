import gymnasium as gym
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DoublePendulumDataset(Dataset):
    def __init__(self, num_sequences=100, seq_len=10, img_size=32):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.img_size = img_size
        self.data = self._generate_data()
        
    def _generate_data(self):
        env = gym.make('Acrobot-v1', render_mode='rgb_array')
        data = []
        for i in range(self.num_sequences):
            env.reset()
            
            # Retour à la position forcée en haut (sans entropie)
            env.unwrapped.state = np.array([3.14, 0.0, 0.0, 0.0], dtype=np.float32)
            # Petite pichenette pour déclencher la chute
            env.step(np.random.choice([0, 2]))

            seq = []
            for _ in range(self.seq_len):
                img = env.render()
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0
                
                # Inversion et seuillage
                img = 1.0 - img
                img[img < 0.2] = 0.0
                
                img = np.transpose(img, (2, 0, 1))
                seq.append(img)
                env.step(1) # Mouvement passif
            data.append(np.array(seq))
            if (i+1) % 50 == 0:
                print(f"Generated {i+1}/{self.num_sequences} sequences")
        return torch.tensor(np.array(data))

    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = DoublePendulumDataset(num_sequences=10, seq_len=5)
    print("Dataset shape:", dataset.data.shape)
