import torch
import matplotlib.pyplot as plt
from .dataset import DoublePendulumDataset
from .model import PhysicsWorldModel
import numpy as np
import gymnasium as gym
import cv2

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 32
    in_channels = 9
    latent_dim = 8

    # Génération d'un cas aléatoire
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    env.reset()
    env.unwrapped.state = np.random.uniform(-2, 2, size=(4,)).astype(np.float32)
    
    full_seq = []
    for _ in range(12): # Pour avoir 10 prédictions (T+2 à T+11)
        img = env.render()
        img = cv2.resize(img, (img_size, img_size))
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img[img < 0.2] = 0.0
        full_seq.append(np.transpose(img, (2, 0, 1)))
        env.step(1)
    batch = torch.tensor(np.array(full_seq)).unsqueeze(0).to(device)

    model = PhysicsWorldModel(in_channels=in_channels, out_channels=3, latent_dim=latent_dim).to(device)
    try:
        model.load_state_dict(torch.load('physics_wm_weights.pth', map_location=device))
        print("Poids chargés.")
    except:
        print("Poids non trouvés.")
    model.eval()

    # Input: Frames 0, 1, 2
    x_in = torch.cat([batch[:, 0], batch[:, 1], batch[:, 2]], dim=1)
    # t pour 10 pas (de la frame 2 à 11)
    t = torch.linspace(0., (12 - 3) * 0.2, 10).to(device)

    with torch.no_grad():
        preds_logits, _, _ = model(x_in, t)
        preds = torch.sigmoid(preds_logits)

    # Affichage des 10 prédictions
    num_preds = 10
    fig, axes = plt.subplots(2, num_preds, figsize=(18, 5))
    
    for i in range(num_preds):
        # GT
        gt = batch[0, i+2].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(gt)
        axes[0, i].set_title(f"GT T+{i+2}")
        axes[0, i].axis('off')

        # Pred
        pred = preds[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(pred)
        axes[1, i].set_title(f"Pred")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('physics_wm_result.png')
    
    # --- Tracé continu du bout du pendule ---
    def find_tip_by_color(img_tensor):
        img = img_tensor.cpu().permute(1, 2, 0).numpy()
        blue_mask = (img[:, :, 2] > 0.1) & (img[:, :, 0] < 0.2)
        coords = np.argwhere(blue_mask)
        if len(coords) < 2: return None
        fixed_anchor = np.array([10, img_size // 2])
        dists = np.sum((coords - fixed_anchor) ** 2, axis=1)
        tip = coords[np.argmax(dists)]
        return tip[1], tip[0]

    gt_path, pred_path = [], []
    for i in range(num_preds):
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
    plt.title("Trajectoire (T+2 à T+11)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('physics_wm_trajectory.png')
    
    print("Visualisation (T+11) et trajectoire sauvegardées.")

if __name__ == '__main__':
    visualize()
