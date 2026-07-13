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
        self.data = self._generate_data()
        
    def _generate_data(self):
        env = gym.make('Pendulum-v1', render_mode='rgb_array')
        data = []
        for i in range(self.num_sequences):
            # Rejection sampling to ensure the pendulum has enough mechanical energy to swing actively
            while True:
                env.reset()
                theta = np.random.uniform(-np.pi, np.pi)
                theta_dot = np.random.uniform(-8.0, 8.0)
                
                # Mechanical energy: E = 0.5 * theta_dot^2 - 10 * cos(theta)
                energy = 0.5 * (theta_dot ** 2) - 10.0 * np.cos(theta)
                # Filter out low-energy states (E < -5.0) where the pendulum barely moves
                if energy > -5.0:
                    env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)
                    break

            seq = []
            for _ in range(self.seq_len):
                img = env.render()
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img.astype(np.float32) / 255.0
                
                # Invert colors (black background) and apply thresholding
                img = 1.0 - img
                img[img < 0.2] = 0.0
                
                img = np.transpose(img, (2, 0, 1))
                seq.append(img)
                env.step(np.array([0.0], dtype=np.float32)) # Passive step (0 torque)
            data.append(np.array(seq))
            if (i+1) % 50 == 0:
                print(f"Generated {i+1}/{self.num_sequences} sequences")
        return torch.tensor(np.array(data))

    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = SimplePendulumDataset(num_sequences=10, seq_len=5)
    print("Dataset shape:", dataset.data.shape)
