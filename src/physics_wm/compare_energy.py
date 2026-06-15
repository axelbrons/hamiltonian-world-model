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
        # Define forward function for odeint
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

# Generate data with ground truth physical states
def generate_data_with_states(num_sequences=300, seq_len=10, img_size=32):
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    images = []
    states = []
    for i in range(num_sequences):
        env.reset()
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        theta1_dot = np.random.uniform(-1.0, 1.0)
        theta2_dot = np.random.uniform(-1.0, 1.0)
        env.unwrapped.state = np.array([theta1, theta2, theta1_dot, theta2_dot], dtype=np.float32)
        
        seq_imgs = []
        seq_states = []
        for _ in range(seq_len):
            img = env.render()
            img = cv2.resize(img, (img_size, img_size))
            img = 1.0 - (img.astype(np.float32) / 255.0)
            img[img < 0.2] = 0.0
            img = np.transpose(img, (2, 0, 1))
            seq_imgs.append(img)
            
            # Map state [theta1, theta2, dtheta1, dtheta2] to 6D [cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]
            s = env.unwrapped.state
            state_6d = [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]]
            seq_states.append(state_6d)
            env.step(1)
        images.append(seq_imgs)
        states.append(seq_states)
        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_sequences} sequences")
    return torch.tensor(np.array(images)).to(device), torch.tensor(np.array(states)).to(device)

# Compute true physical energy of the Acrobot from the 6D state
def acrobot_energy_6d(y):
    # y is shape [..., 6]
    cos_t1, sin_t1, cos_t2, sin_t2, dtheta1, dtheta2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3], y[..., 4], y[..., 5]
    m11 = 3.5 + cos_t2
    m22 = 1.25
    m12 = 1.25 + 0.5 * cos_t2
    T = 0.5 * (m11 * dtheta1**2 + 2.0 * m12 * dtheta1 * dtheta2 + m22 * dtheta2**2)
    cos_t1_t2 = cos_t1 * cos_t2 - sin_t1 * sin_t2
    V = - 9.8 * (1.5 * cos_t1 + 0.5 * cos_t1_t2)
    return T + V

def main():
    print("Generating training data...")
    train_imgs, train_states = generate_data_with_states(num_sequences=250, seq_len=10)
    
    latent_dim = 8
    seq_len = 10
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
    t_train = torch.linspace(0., (seq_len - 3) * dt, seq_len - 2).to(device)
    
    base_model.train()
    for epoch in range(15):
        epoch_loss = 0.0
        # Simple batching (batch size 16)
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

    # 3. Train Regressors mapping latent z -> 6D physical state
    print("Training State Regressors...")
    hnn_regressor = nn.Sequential(
        nn.Linear(latent_dim * 2, 64),
        nn.ReLU(),
        nn.Linear(64, 6)
    ).to(device)
    base_regressor = nn.Sequential(
        nn.Linear(latent_dim * 2, 64),
        nn.ReLU(),
        nn.Linear(64, 6)
    ).to(device)
    
    opt_hnn_reg = optim.Adam(hnn_regressor.parameters(), lr=1e-3)
    opt_base_reg = optim.Adam(base_regressor.parameters(), lr=1e-3)
    
    # Collect training pairs for regression
    hnn_model.eval()
    base_model.eval()
    with torch.no_grad():
        x_in_all = torch.cat([train_imgs[:, 0], train_imgs[:, 1], train_imgs[:, 2]], dim=1)
        # HNN latents
        _, _, z_hnn = hnn_model(x_in_all, t_train) # Shape: [seq_len-2, batch, 16]
        z_hnn = z_hnn.transpose(0, 1) # Shape: [batch, seq_len-2, 16]
        
        # Baseline latents
        _, _, z_base = base_model(x_in_all, t_train) # Shape: [seq_len-2, batch, 16]
        z_base = z_base.transpose(0, 1) # Shape: [batch, seq_len-2, 16]
        
    # Ground truth states corresponding to t = 2 to seq_len-1
    states_target = train_states[:, 2:] # Shape: [batch, seq_len-2, 6]
    
    # Flatten for regression training
    z_hnn_flat = z_hnn.reshape(-1, latent_dim * 2)
    z_base_flat = z_base.reshape(-1, latent_dim * 2)
    states_flat = states_target.reshape(-1, 6)
    
    # Train regressors
    for reg_epoch in range(50):
        # HNN Regressor step
        opt_hnn_reg.zero_grad()
        pred_hnn_states = hnn_regressor(z_hnn_flat)
        loss_hnn = nn.MSELoss()(pred_hnn_states, states_flat)
        loss_hnn.backward()
        opt_hnn_reg.step()
        
        # Baseline Regressor step
        opt_base_reg.zero_grad()
        pred_base_states = base_regressor(z_base_flat)
        loss_base = nn.MSELoss()(pred_base_states, states_flat)
        loss_base.backward()
        opt_base_reg.step()
        
        if (reg_epoch+1) % 10 == 0:
            print(f"Regressor Epoch {reg_epoch+1}/50, HNN MSE: {loss_hnn.item():.4f}, Base MSE: {loss_base.item():.4f}")

    # 4. Long rollout evaluation (e.g. 40 steps) on a test sequence
    print("Evaluating long rollout on a test trajectory...")
    test_imgs, test_states = generate_data_with_states(num_sequences=1, seq_len=45)
    
    rollout_len = 40
    t_eval = torch.linspace(0., rollout_len * dt, rollout_len).to(device)
    
    x_in_test = torch.cat([test_imgs[:, 0], test_imgs[:, 1], test_imgs[:, 2]], dim=1)
    
    with torch.no_grad():
        # HNN Rollout
        _, _, z_hnn_test = hnn_model(x_in_test, t_eval) # [40, 1, 16]
        z_hnn_test = z_hnn_test.squeeze(1) # [40, 16]
        pred_hnn_states_eval = hnn_regressor(z_hnn_test) # [40, 6]
        
        # Baseline Rollout
        _, _, z_base_test = base_model(x_in_test, t_eval) # [40, 1, 16]
        z_base_test = z_base_test.squeeze(1) # [40, 16]
        pred_base_states_eval = base_regressor(z_base_test) # [40, 6]
        
    # True energies
    true_states_eval = test_states[0, 2:2+rollout_len] # [40, 6]
    true_energies = acrobot_energy_6d(true_states_eval).cpu().numpy()
    
    # Predicted energies
    hnn_energies = acrobot_energy_6d(pred_hnn_states_eval).cpu().numpy()
    base_energies = acrobot_energy_6d(pred_base_states_eval).cpu().numpy()
    
    # 5. Plotting results
    print("Plotting energy comparison...")
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(rollout_len) * dt
    
    plt.plot(time_steps, true_energies, 'g-', label='Vérité Terrain (Ground Truth)', linewidth=2.5)
    plt.plot(time_steps, hnn_energies, 'b--', label='Notre Modèle (HNN-ODE)', linewidth=2.0)
    plt.plot(time_steps, base_energies, 'r:', label='Baseline (Neural ODE classique)', linewidth=2.0)
    
    plt.xlabel('Temps (s)', fontsize=12)
    plt.ylabel('Énergie Totale (J)', fontsize=12)
    plt.title("Comparaison de la conservation de l'énergie physique au cours d'un long rollout", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    output_path = 'physics_wm_energy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Energy comparison plot saved to {output_path}")

if __name__ == '__main__':
    main()
