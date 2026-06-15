import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_sequences} sequences")
    return torch.tensor(np.array(images)).to(device)

def main():
    print("Generating training data...")
    train_imgs = generate_sequences(num_sequences=250, seq_len=10)
    
    latent_dim = 8
    dt = 0.2
    
    # 1. Load HNN-ODE Model
    print("Loading trained HNN-ODE Model...")
    hnn_model = PhysicsWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    if os.path.exists('physics_wm_weights.pth'):
        hnn_model.load_state_dict(torch.load('physics_wm_weights.pth', map_location=device))
        print("Loaded weights from physics_wm_weights.pth")
    else:
        print("ERROR: physics_wm_weights.pth not found. Please train the model first.")
        return

    # 2. Train Baseline Model (Standard Neural ODE)
    print("Training Baseline World Model...")
    base_model = BaselineWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    optimizer_base = optim.Adam(base_model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    t_train = torch.linspace(0., 7 * dt, 8).to(device)
    
    base_model.train()
    for epoch in range(15):
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
        print(f"Baseline Epoch {epoch+1}/15, Loss: {epoch_loss/(train_imgs.size(0)/16):.4f}")

    # 3. Generate test sequences with length 22 (to get 20 prediction steps)
    test_len = 22
    pred_len = 20
    print(f"Generating {100} test sequences of length {test_len}...")
    test_imgs = generate_sequences(num_sequences=100, seq_len=test_len)
    
    hnn_model.eval()
    base_model.eval()
    
    x_in_test = torch.cat([test_imgs[:, 0], test_imgs[:, 1], test_imgs[:, 2]], dim=1)
    targets_test = test_imgs[:, 2:].transpose(0, 1) # Shape: [20, 100, 3, 32, 32]
    
    t_eval = torch.linspace(0., (pred_len - 1) * dt, pred_len).to(device)
    
    print("Evaluating reconstruction errors on test set...")
    with torch.no_grad():
        # HNN predictions
        preds_hnn_logits, _, _ = hnn_model(x_in_test, t_eval)
        preds_hnn = torch.sigmoid(preds_hnn_logits) # [20, 100, 3, 32, 32]
        
        # Baseline predictions
        preds_base_logits, _, _ = base_model(x_in_test, t_eval)
        preds_base = torch.sigmoid(preds_base_logits) # [20, 100, 3, 32, 32]
        
    # Compute MSE for each timestep
    mse_hnn = []
    mse_base = []
    
    for t_idx in range(pred_len):
        # MSE over pixels (dim: channels, H, W) and batch
        err_hnn = ((preds_hnn[t_idx] - targets_test[t_idx])**2).mean().item()
        err_base = ((preds_base[t_idx] - targets_test[t_idx])**2).mean().item()
        
        mse_hnn.append(err_hnn)
        mse_base.append(err_base)
        
    # Plot results
    print("Plotting reconstruction error comparison...")
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(2, 2 + pred_len)
    
    plt.plot(time_steps, mse_hnn, 'b-o', label='Notre Modèle (HNN-ODE)', linewidth=2.0, markersize=5)
    plt.plot(time_steps, mse_base, 'r-x', label='Baseline (Neural ODE classique)', linewidth=2.0, markersize=5)
    
    plt.xlabel('Pas de temps (T+t)', fontsize=12)
    plt.ylabel('Erreur de reconstruction moyenne (MSE)', fontsize=12)
    plt.title("Évolution temporelle de l'erreur de reconstruction sur un long horizon", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    output_path = 'physics_wm_reconstruction_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Reconstruction comparison plot saved to {output_path}")

if __name__ == '__main__':
    main()
