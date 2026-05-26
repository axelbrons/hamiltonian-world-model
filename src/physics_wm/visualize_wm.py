import torch
import matplotlib.pyplot as plt
from .dataset import DoublePendulumDataset
from .model import PhysicsWorldModel
import numpy as np

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 10
    latent_dim = 8
    img_size = 64

    # Charger un petit dataset de test
    dataset = DoublePendulumDataset(num_sequences=5, seq_len=seq_len, img_size=img_size)
    
    # Initialiser le modèle et charger les poids entraînés
    model = PhysicsWorldModel(in_channels=6, out_channels=3, latent_dim=latent_dim).to(device)
    try:
        model.load_state_dict(torch.load('physics_wm_weights.pth', map_location=device))
        print("Poids chargés avec succès.")
    except FileNotFoundError:
        print("Fichier de poids non trouvé, utilisation d'un modèle aléatoire.")
    model.eval()

    # Prendre une séquence
    batch = dataset[0].unsqueeze(0).to(device) # (1, seq_len, 3, 32, 32)
    x_in = torch.cat([batch[:, 0], batch[:, 1]], dim=1)
    t = torch.linspace(0., (seq_len - 2) * 0.1, seq_len - 1).to(device)

    with torch.no_grad():
        preds, _, _ = model(x_in, t)

    # Affichage
    fig, axes = plt.subplots(2, seq_len - 1, figsize=(15, 5))
    for i in range(seq_len - 1):
        # Ground Truth (à partir du frame 1)
        gt = batch[0, i+1].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(gt)
        axes[0, i].set_title(f"GT T+{i+1}")
        axes[0, i].axis('off')

        # Prediction
        pred = preds[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(pred)
        axes[1, i].set_title(f"Pred T+{i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('physics_wm_result.png')
    print("Résultat sauvegardé dans physics_wm_result.png")

if __name__ == '__main__':
    visualize()
