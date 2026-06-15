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
    latent_dim = 1

    # Generate a single test sequence using Gymnasium Pendulum-v1 with active control
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    env.reset()
    env.unwrapped.state = np.random.uniform(-2, 2, size=(2,)).astype(np.float32)
    
    seq_len = 66 # 3 initial frames + 9 blocks of 7 predictions (with overlap) = 66 frames total
    full_seq = []
    full_acts = []
    for _ in range(seq_len):
        img = env.render()
        img = cv2.resize(img, (img_size, img_size))
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img[img < 0.2] = 0.0
        full_seq.append(np.transpose(img, (2, 0, 1)))
        
        # Apply random torques/impulses
        act = np.random.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)
        full_acts.append(act)
        env.step(act)
    batch = torch.tensor(np.array(full_seq)).unsqueeze(0).to(device)
    batch_acts = torch.tensor(np.array(full_acts)).unsqueeze(0).to(device) # [1, 66, 1]

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
    num_blocks = 9
    num_steps = seq_len - 2 # 64 predictions
    
    preds_all = []
    current_x = torch.cat([batch[:, 0], batch[:, 1], batch[:, 2]], dim=1) # [1, 9, 32, 32]
    t_block = torch.linspace(0., (block_len - 1) * 0.2, block_len).to(device)
    
    with torch.no_grad():
        for b in range(num_blocks):
            # Extract actions for the current block rollout
            start_idx = b * 7 + 2
            actions_block = batch_acts[:, start_idx : start_idx + 7].transpose(0, 1) # [7, 1, 1]
            
            # Predict block_len frames from current context and actions
            preds_block_logits, _, _ = model(current_x, actions_block, t_block)
            preds_block = torch.sigmoid(preds_block_logits)
            
            # Slice to avoid duplicates at block boundaries
            if b == 0:
                preds_all.append(preds_block) # [8, 1, 3, 32, 32]
            else:
                preds_all.append(preds_block[1:]) # [7, 1, 3, 32, 32]
                
            if b < num_blocks - 1:
                # Recalibrate: take last 3 predicted frames of the current block
                frame_first = preds_block[-3]
                frame_second = preds_block[-2]
                frame_third = preds_block[-1]
                
                # Stack them to form the input for the next block
                current_x = torch.cat([frame_first, frame_second, frame_third], dim=1)
                
        preds = torch.cat(preds_all, dim=0) # [64, 1, 3, 32, 32]


    # Plot 10 evenly-spaced predictions across the entire sequence
    num_cols = 10
    indices_to_show = np.linspace(0, num_steps - 1, num_cols, dtype=int)
    fig, axes = plt.subplots(2, num_cols, figsize=(18, 5))
    
    for idx, i in enumerate(indices_to_show):
        # GT
        gt = batch[0, i+2].cpu().permute(1, 2, 0).numpy()
        axes[0, idx].imshow(gt)
        axes[0, idx].set_title(f"GT T+{i+2}")
        axes[0, idx].axis('off')

        # Pred
        pred = preds[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, idx].imshow(pred)
        axes[1, idx].set_title(f"Pred")
        axes[1, idx].axis('off')

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

if __name__ == '__main__':
    visualize()
