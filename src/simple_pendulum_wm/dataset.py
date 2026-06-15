import gymnasium as gym
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class SimplePendulumDataset(Dataset):
    def __init__(self, num_sequences=100, seq_len=20, img_size=32):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.img_size = img_size
        self.data, self.actions = self._generate_data()
        
    def _generate_data(self):
        env = gym.make('Pendulum-v1', render_mode='rgb_array')
        data = []
        actions = []
        for i in range(self.num_sequences):
            env.reset()
            
            # Random initial state for simple pendulum: theta in [-pi, pi], theta_dot in [-2.0, 2.0]
            theta = np.random.uniform(-np.pi, np.pi)
            theta_dot = np.random.uniform(-2.0, 2.0)
            
            env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)

            seq = []
            seq_acts = []
            for _ in range(self.seq_len):
                img = env.render()
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0
                
                # Invert colors (black background) and apply thresholding
                img = 1.0 - img
                img[img < 0.2] = 0.0
                
                img = np.transpose(img, (2, 0, 1))
                seq.append(img)
                
                # Random active torque in [-2.0, 2.0]
                act = np.random.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)
                seq_acts.append(act)
                env.step(act)
            data.append(np.array(seq))
            actions.append(np.array(seq_acts))
            if (i+1) % 50 == 0:
                print(f"Generated {i+1}/{self.num_sequences} sequences")
        return torch.tensor(np.array(data)), torch.tensor(np.array(actions))

    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        return self.data[idx], self.actions[idx]

if __name__ == "__main__":
    dataset = SimplePendulumDataset(num_sequences=10, seq_len=5)
    print("Dataset shape:", dataset.data.shape)
    print("Actions shape:", dataset.actions.shape)
