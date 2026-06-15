import torch
import torch.nn as nn
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

# Generate a single long sequence of frames
def generate_long_sequence(seq_len=150, img_size=32):
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    env.reset()
    
    # Random initial state for interesting chaotic/oscillatory motion
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
        env.step(1) # Passive step
    return torch.tensor(np.array(seq_imgs)).unsqueeze(0).to(device) # Shape: [1, seq_len, 3, 32, 32]

def main():
    latent_dim = 8
    dt = 0.2
    seq_len = 150
    
    # 1. Load HNN-ODE Model
    print("Loading trained HNN-ODE Model...")
    hnn_model = PhysicsWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    if os.path.exists('physics_wm_weights.pth'):
        hnn_model.load_state_dict(torch.load('physics_wm_weights.pth', map_location=device))
        print("Loaded weights from physics_wm_weights.pth")
    else:
        print("ERROR: physics_wm_weights.pth not found. Please train the model first.")
        return
    
    hnn_model.eval()
    
    # 2. Generate long test sequence
    print(f"Generating a test sequence of length {seq_len}...")
    seq_imgs = generate_long_sequence(seq_len=seq_len) # [1, 150, 3, 32, 32]
    
    x_in = torch.cat([seq_imgs[:, 0], seq_imgs[:, 1], seq_imgs[:, 2]], dim=1) # [1, 9, 32, 32]
    
    # Time vector for rollout
    t = torch.linspace(0., (seq_len - 3) * dt, seq_len - 2).to(device)
    
    # 3. Rollout in latent space
    print("Running latent rollout...")
    with torch.no_grad():
        _, _, z_t = hnn_model(x_in, t) # z_t has shape [seq_len-2, 1, 16]
        z_t = z_t.squeeze(1) # [seq_len-2, 16]
        
    # Extract first dimension of position (q0) and moment (p0)
    # q is first half of latent dimension (index 0 to 7)
    # p is second half of latent dimension (index 8 to 15)
    q0 = z_t[:, 0].cpu().numpy()
    p0 = z_t[:, latent_dim].cpu().numpy()
    
    # 4. Plotting the phase space
    print("Plotting phase space...")
    plt.figure(figsize=(8, 8))
    
    # Trace the line representing the trajectory
    plt.plot(q0, p0, 'k-', alpha=0.3, label='Trajectoire latente')
    
    # Scatter plot with color coding by time step to see the trajectory direction
    colors = np.arange(len(q0))
    sc = plt.scatter(q0, p0, c=colors, cmap='plasma', s=30, edgecolor='none', zorder=3)
    
    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Étapes temporelles (t)', fontsize=12)
    
    plt.xlabel('$q_0$ (Coordonnée de position latente)', fontsize=12)
    plt.ylabel('$p_0$ (Coordonnée de moment latente)', fontsize=12)
    plt.title("Espace des phases latent prédit par le HNN-ODE ($p_0$ vs $q_0$)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add annotation for start and end points
    plt.scatter(q0[0], p0[0], color='green', marker='o', s=100, label='Début (t=0)', zorder=5)
    plt.scatter(q0[-1], p0[-1], color='red', marker='x', s=100, label='Fin', zorder=5)
    plt.legend()
    
    output_path = 'physics_wm_latent_phase_space.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Latent phase space plot saved to {output_path}")

if __name__ == '__main__':
    main()
