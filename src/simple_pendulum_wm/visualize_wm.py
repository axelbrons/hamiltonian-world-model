import torch
import matplotlib.pyplot as plt
from .dataset import SimplePendulumDataset
from .model import PhysicsWorldModel
import numpy as np
import gymnasium as gym
import cv2
import os

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 32
    in_channels = 9
    latent_dim = 2

    # Generate a single test sequence using Gymnasium Pendulum-v1 (passive, with friction)
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env.reset()
    
    # Initialize the pendulum at a random state
    theta = np.random.uniform(-np.pi, np.pi)
    theta_dot = np.random.uniform(-8.0, 8.0)
    env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)
    
    seq_len = 150 # 3 initial frames + 21 blocks of 7 predictions (with overlap) = 150 frames total
    full_seq = []
    for _ in range(seq_len):
        img = env.render()
        img = cv2.resize(img, (img_size, img_size))
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img[img < 0.2] = 0.0
        full_seq.append(np.transpose(img, (2, 0, 1)))
        env.step(np.array([0.0], dtype=np.float32)) # Passive step (friction will slow it down)
    batch = torch.tensor(np.array(full_seq)).unsqueeze(0).to(device)

    model = PhysicsWorldModel(in_channels=in_channels, out_channels=3, latent_dim=latent_dim).to(device)
    if os.path.exists('simple_pendulum_weights.pth'):
        model.load_state_dict(torch.load('simple_pendulum_weights.pth', map_location=device))
        print("Loaded weights from simple_pendulum_weights.pth.")
    else:
        print("ERROR: simple_pendulum_weights.pth not found. Please train the model first.")
        return
        
    model.eval()

    # Closed-loop autoregressive block rollout (recalibrating latent space via decoder-encoder loop)
    block_len = 8
    num_blocks = 21
    num_steps = seq_len - 2 # 148 predictions
    
    preds_all = []
    z_all = []
    current_x = torch.cat([batch[:, 0], batch[:, 1], batch[:, 2]], dim=1) # [1, 9, 32, 32]
    t_block = torch.linspace(0., (block_len - 1) * 0.2, block_len).to(device)
    
    with torch.no_grad():
        for b in range(num_blocks):
            # Predict block_len frames from current context (passive Leapfrog integration)
            preds_block_logits, _, z_block = model(current_x, t_block)
            preds_block = torch.sigmoid(preds_block_logits)
            
            # Slice to avoid duplicates at block boundaries
            if b == 0:
                preds_all.append(preds_block) # [8, 1, 3, 32, 32]
                z_all.append(z_block) # [8, 1, 2]
            else:
                preds_all.append(preds_block[1:]) # [7, 1, 3, 32, 32]
                z_all.append(z_block[1:]) # [7, 1, 2]
                
            if b < num_blocks - 1:
                # Recalibrate: take last 3 predicted frames of the current block
                frame_first = preds_block[-3]
                frame_second = preds_block[-2]
                frame_third = preds_block[-1]
                
                # Apply thresholding to remove sigmoid background blur/glow and match encoder's training distribution
                frame_first = torch.where(frame_first < 0.2, torch.zeros_like(frame_first), frame_first)
                frame_second = torch.where(frame_second < 0.2, torch.zeros_like(frame_second), frame_second)
                frame_third = torch.where(frame_third < 0.2, torch.zeros_like(frame_third), frame_third)
                
                # Stack them to form the input for the next block
                current_x = torch.cat([frame_first, frame_second, frame_third], dim=1)
                
        preds = torch.cat(preds_all, dim=0) # [148, 1, 3, 32, 32]

    # Plot dense grid: T=1 to T=50 (5 blocks of 10 steps each, with GT and Pred rows paired)
    num_steps_to_plot = 50
    steps_per_row = 10
    num_blocks_plot = num_steps_to_plot // steps_per_row  # 5 blocks
    
    fig, axes = plt.subplots(num_blocks_plot * 2, steps_per_row, figsize=(15, 12))
    
    for block in range(num_blocks_plot):
        for col in range(steps_per_row):
            t_step = block * steps_per_row + col + 1  # 1 to 50
            i = t_step - 1  # prediction index
            
            # Row index for GT and Pred
            gt_row = block * 2
            pred_row = block * 2 + 1
            
            # GT (first prediction starts at batch index 2)
            gt = batch[0, i+2].cpu().permute(1, 2, 0).numpy()
            axes[gt_row, col].imshow(gt)
            if col == 0:
                axes[gt_row, col].set_ylabel("GT", fontsize=12, fontweight='bold')
            axes[gt_row, col].set_title(f"T={t_step}", fontsize=8)
            axes[gt_row, col].set_xticks([])
            axes[gt_row, col].set_yticks([])
            
            # Pred
            pred = preds[i, 0].cpu().permute(1, 2, 0).numpy()
            axes[pred_row, col].imshow(pred)
            if col == 0:
                axes[pred_row, col].set_ylabel("Pred", fontsize=12, fontweight='bold')
            axes[pred_row, col].set_xticks([])
            axes[pred_row, col].set_yticks([])
            
    plt.tight_layout()
    plt.savefig('simple_pendulum_result.png')
    print("Reconstruction results saved to simple_pendulum_result.png.")
    
    # --- Track the tip of the pendulum ---
    def find_tip_by_color(img_tensor):
        img = img_tensor.cpu().permute(1, 2, 0).numpy()
        # Cyan color in inverted image: low red, high green and blue
        cyan_mask = (img[:, :, 1] > 0.1) & (img[:, :, 2] > 0.1) & (img[:, :, 0] < 0.2)
        coords = np.argwhere(cyan_mask)
        if len(coords) < 2: return None
        fixed_anchor = np.array([img_size // 2, img_size // 2])
        dists = np.sum((coords - fixed_anchor) ** 2, axis=1)
        tip = coords[np.argmax(dists)]
        return tip[1], tip[0]

    gt_path, pred_path = [], []
    for i in range(num_steps):
        g = find_tip_by_color(batch[0, i+2])
        p = find_tip_by_color(preds[i, 0])
        if g: gt_path.append(g)
        if p: pred_path.append(p)
    
    gt_path, pred_path = np.array(gt_path), np.array(pred_path)
    
    plt.figure(figsize=(6, 6))
    if len(gt_path) > 0:
        plt.plot(gt_path[:, 0], gt_path[:, 1], 'g-', label='Vérité Terrain', lw=2, alpha=0.8)
        plt.scatter(gt_path[-1, 0], gt_path[-1, 1], c='g', s=60, zorder=5)
    if len(pred_path) > 0:
        plt.plot(pred_path[:, 0], pred_path[:, 1], 'r--', label='Prédiction HNN', lw=2)
        plt.scatter(pred_path[-1, 0], pred_path[-1, 1], c='r', marker='x', s=60, zorder=5)
    
    plt.xlim(0, img_size); plt.ylim(img_size, 0)
    plt.title(f"Trajectoire du bout du pendule (T+2 à T+{num_steps+1})")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('simple_pendulum_trajectory.png')
    print("Trajectory results saved to simple_pendulum_trajectory.png.")

    # --- Plot Latent Space (q1 vs q2 and q1 vs p1) ---
    z_all_tensor = torch.cat(z_all, dim=0) # [num_steps, 1, 2*latent_dim]
    q1 = z_all_tensor[:, 0, 0].cpu().numpy()
    q2 = z_all_tensor[:, 0, 1].cpu().numpy()
    p1 = z_all_tensor[:, 0, 2].cpu().numpy()
    p2 = z_all_tensor[:, 0, 3].cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Configuration space (q1 vs q2) -> Cartesien (x, y) = (sin theta, -cos theta)
    axes[0].plot(q1, q2, 'b-', label='Trajectoire Latente (q1, q2)', alpha=0.8)
    axes[0].scatter(q1[0], q2[0], c='green', marker='o', s=80, label='Début (t=0)', zorder=5)
    axes[0].scatter(q1[-1], q2[-1], c='red', marker='x', s=80, label='Fin (t=T)', zorder=5)
    step = max(1, len(q1) // 8)
    for idx in range(0, len(q1) - 1, step):
        axes[0].annotate('', xy=(q1[idx+1], q2[idx+1]), xytext=(q1[idx], q2[idx]),
                         arrowprops=dict(arrowstyle="->", color='blue', lw=1.5))
    axes[0].set_xlabel('Position Latente q1')
    axes[0].set_ylabel('Position Latente q2')
    axes[0].set_title('Espace de Configuration (q1 vs q2)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # 2. Phase space (q1 vs p1)
    axes[1].plot(q1, p1, 'm-', label='Espace des phases (q1, p1)', alpha=0.8)
    axes[1].scatter(q1[0], p1[0], c='green', marker='o', s=80, label='Début (t=0)', zorder=5)
    axes[1].scatter(q1[-1], p1[-1], c='red', marker='x', s=80, label='Fin (t=T)', zorder=5)
    for idx in range(0, len(q1) - 1, step):
        axes[1].annotate('', xy=(q1[idx+1], p1[idx+1]), xytext=(q1[idx], p1[idx]),
                         arrowprops=dict(arrowstyle="->", color='purple', lw=1.5))
    axes[1].set_xlabel('Position Latente q1')
    axes[1].set_ylabel('Moment Latent p1')
    axes[1].set_title('Espace des Phases Latent (q1 vs p1)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_pendulum_latent_phase_space.png')
    print("Latent phase space portrait saved to simple_pendulum_latent_phase_space.png.")

if __name__ == '__main__':
    visualize()
