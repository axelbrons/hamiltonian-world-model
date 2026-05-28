import torch
import matplotlib.pyplot as plt
from .dataset import DoublePendulumDataset
from .model import PhysicsWorldModel
import numpy as np

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 10
    latent_dim = 8
    img_size = 32
    in_channels = 9

    dataset = DoublePendulumDataset(num_sequences=5, seq_len=seq_len, img_size=img_size)
    model = PhysicsWorldModel(in_channels=in_channels, out_channels=3, latent_dim=latent_dim).to(device)
    
    try:
        model.load_state_dict(torch.load('physics_wm_weights.pth', map_location=device))
        print("Poids chargés.")
    except:
        print("Poids non trouvés.")
    
    model.eval()

    batch = dataset[0].unsqueeze(0).to(device)
    x_in = torch.cat([batch[:, 0], batch[:, 1], batch[:, 2]], dim=1)
    t = torch.linspace(0., (seq_len - 3) * 0.2, seq_len - 2).to(device)

    with torch.no_grad():
        preds_logits, _, _ = model(x_in, t)
        preds = torch.sigmoid(preds_logits)

    # Retour à un affichage 2 lignes : GT et Pred
    num_preds = seq_len - 2
    fig, axes = plt.subplots(2, num_preds, figsize=(15, 5))
    
    for i in range(num_preds):
        gt = batch[0, i+2].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(gt)
        axes[0, i].set_title(f"GT T+{i+2}")
        axes[0, i].axis('off')

        pred = preds[i, 0].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(pred)
        axes[1, i].set_title(f"Pred")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('physics_wm_result.png')
    print("Résultat sauvegardé.")

if __name__ == '__main__':
    visualize()
