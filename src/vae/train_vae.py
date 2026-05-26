import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.vae.vae import VAE, loss_function
from src.vae.dataloader import GymDataset

# 1. Paramètres
DATA_DIR = "carracing_dataset"
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_dir):
    """Charge toutes les images de tous les rollouts."""
    files = glob.glob(os.path.join(data_dir, "rollout_*.npz"))
    all_frames = []
    
    print(f"Chargement de {len(files)} fichiers de données...")
    for f in files:
        with np.load(f) as data:
            all_frames.append(data['obs'])
            
    # On concatène tout en un seul gros tableau numpy
    return np.concatenate(all_frames, axis=0)

def train():
    # 2. Préparation des données
    obs_data = load_data(DATA_DIR)
    print(f"Nombre total d'images : {len(obs_data)}")
    
    # On utilise le GymDataset que tu as défini
    dataset = GymDataset(obs_data)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialisation du modèle
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Boucle d'entraînement
    print(f"Début de l'entraînement sur {DEVICE}...")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch.to(DEVICE) # Shape: (B, 64, 64, 3)
            
            # Forward pass
            reconst, mu, log_var = model(batch)
            
            # Calcul de la perte
            loss = loss_function(reconst, batch, mu, log_var)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()/BATCH_SIZE:.4f}")
        
        avg_loss = total_loss / len(dataset)
        print(f"===> Epoch {epoch+1} terminée. Perte moyenne : {avg_loss:.4f}")
        
        # Sauvegarde du modèle à chaque epoch
        torch.save(model.state_dict(), "src/vae/vae_weights.pth")

    print("Entraînement terminé ! Modèle sauvegardé sous 'src/vae/vae_weights.pth'")

if __name__ == "__main__":
    train()
