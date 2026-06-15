import torch
import torch.nn as nn
from torchdiffeq import odeint
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import local classes
from src.physics_wm.encoder_decoder import Encoder, Decoder
from src.physics_wm.hnn import HNN, HNN_ODE
from src.physics_wm.model import PhysicsWorldModel

# Define Baseline World Model (Neural ODE without Hamiltonian constraints)
class BaselineWorldModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(BaselineWorldModel, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)
        
        # Standard MLP for Neural ODE dynamics
        self.ode_func = nn.Sequential(
            nn.Linear(latent_dim * 2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, latent_dim * 2)
        )
        
    def forward(self, x, t):
        z0 = self.encoder(x)
        class ODEFuncWrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
            def forward(self, t, z):
                return self.net(z)
                
        z_t = odeint(ODEFuncWrapper(self.ode_func), z0, t, method='rk4', options={'step_size': 0.05})
        seq_len, batch_size, latent_dim_2 = z_t.shape
        z_t_flat = z_t.view(-1, latent_dim_2)
        preds = self.decoder(z_t_flat)
        preds = preds.view(seq_len, batch_size, *preds.shape[1:])
        return preds, z0, z_t

# Generate image data for sequences
def generate_sequences(num_sequences=200, seq_len=10, img_size=32):
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    images = []
    for i in range(num_sequences):
        env.reset()
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        theta1_dot = np.random.uniform(-1.0, 1.0)
        theta2_dot = np.random.uniform(-1.0, 1.0)
        env.unwrapped.state = np.array([theta1, theta2, theta1_dot, theta2_dot], dtype=np.float32)
        
        seq_imgs = []
        for _ in range(seq_len):
            img = env.render()
            img = cv2.resize(img, (img_size, img_size))
            img = 1.0 - (img.astype(np.float32) / 255.0)
            img[img < 0.2] = 0.0
            img = np.transpose(img, (2, 0, 1))
            seq_imgs.append(img)
            env.step(1)
        images.append(seq_imgs)
    return torch.tensor(np.array(images)).to(device)

def main():
    latent_dim = 8
    dt = 0.2
    
    # 1. Load HNN-ODE Model
    print("Loading HNN-ODE Model...")
    hnn_model = PhysicsWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    if os.path.exists('physics_wm_weights.pth'):
        hnn_model.load_state_dict(torch.load('physics_wm_weights.pth', map_location=device))
        print("Loaded weights from physics_wm_weights.pth")
    else:
        print("ERROR: physics_wm_weights.pth not found.")
        return

    # 2. Train Baseline Model quickly to show degradation
    print("Training Baseline Model...")
    train_imgs = generate_sequences(num_sequences=200, seq_len=10)
    base_model = BaselineWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    optimizer_base = optim.Adam(base_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    t_train = torch.linspace(0., 7 * dt, 8).to(device)
    
    base_model.train()
    for epoch in range(12):
        permutation = torch.randperm(train_imgs.size(0))
        for i in range(0, train_imgs.size(0), 16):
            indices = permutation[i:i+16]
            batch_imgs = train_imgs[indices]
            optimizer_base.zero_grad()
            x_in = torch.cat([batch_imgs[:, 0], batch_imgs[:, 1], batch_imgs[:, 2]], dim=1)
            targets = batch_imgs[:, 2:].transpose(0, 1)
            preds, _, _ = base_model(x_in, t_train)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer_base.step()

    # 3. Generate a single test sequence of length 22
    print("Generating a test sequence of length 22...")
    test_seq = generate_sequences(num_sequences=1, seq_len=22) # [1, 22, 3, 32, 32]
    
    hnn_model.eval()
    base_model.eval()
    
    x_in_test = torch.cat([test_seq[:, 0], test_seq[:, 1], test_seq[:, 2]], dim=1)
    
    # 20 steps of prediction: from T+2 to T+21
    t_eval = torch.linspace(0., 19 * dt, 20).to(device)
    
    with torch.no_grad():
        # HNN Predictions
        preds_hnn_logits, _, _ = hnn_model(x_in_test, t_eval)
        preds_hnn = torch.sigmoid(preds_hnn_logits) # [20, 1, 3, 32, 32]
        
        # Baseline Predictions
        preds_base_logits, _, _ = base_model(x_in_test, t_eval)
        preds_base = torch.sigmoid(preds_base_logits) # [20, 1, 3, 32, 32]
        
    # Select 10 timesteps to show: T+2, T+4, T+6, T+8, T+10, T+12, T+14, T+16, T+18, T+20
    indices_to_show = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    num_cols = len(indices_to_show)
    
    fig, axes = plt.subplots(3, num_cols, figsize=(18, 7))
    
    for idx, t_idx in enumerate(indices_to_show):
        # Ground Truth (T+t_idx+2)
        gt_img = test_seq[0, t_idx + 2].cpu().permute(1, 2, 0).numpy()
        axes[0, idx].imshow(gt_img)
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].text(-15, 16, 'Ground Truth', fontsize=12, fontweight='bold', va='center', ha='right')
        axes[0, idx].set_title(f"T+{t_idx+2}")
        
        # HNN-ODE Prediction
        hnn_img = preds_hnn[t_idx, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, idx].imshow(hnn_img)
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].text(-15, 16, 'Notre HNN-ODE', fontsize=12, fontweight='bold', va='center', ha='right')
            
        # Baseline Prediction
        base_img = preds_base[t_idx, 0].cpu().permute(1, 2, 0).numpy()
        axes[2, idx].imshow(base_img)
        axes[2, idx].axis('off')
        if idx == 0:
            axes[2, idx].text(-15, 16, 'Neural ODE\n(sans HNN)', fontsize=12, fontweight='bold', va='center', ha='right')
            
    plt.tight_layout()
    output_path = 'physics_wm_long_rollout.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Long rollout visualization saved to {output_path}")

if __name__ == '__main__':
    main()
