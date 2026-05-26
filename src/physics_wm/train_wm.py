import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import DoublePendulumDataset
from .model import PhysicsWorldModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 50
    batch_size = 16
    learning_rate = 1e-3
    seq_len = 10
    latent_dim = 8 # q and p each have dimension 4 (total 8) for a double pendulum
    
    # Dataset
    print("Generating dataset...")
    dataset = DoublePendulumDataset(num_sequences=200, seq_len=seq_len, img_size=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model: encoder takes 2 frames (6 channels), decoder outputs 1 frame (3 channels)
    model = PhysicsWorldModel(in_channels=6, out_channels=3, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Time steps for ODE
    # We use first 2 frames to encode z_1, then predict frames 1 to seq_len-1
    # Total frames to predict: seq_len - 1
    t = torch.linspace(0., (seq_len - 2) * 0.1, seq_len - 1).to(device)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device) # Shape: (batch, seq_len, 3, 32, 32)
            
            optimizer.zero_grad()
            
            # Input to encoder: first two frames
            # Shape: (batch, 6, 32, 32)
            x_in = torch.cat([batch[:, 0], batch[:, 1]], dim=1)
            
            # Targets: frames 1 to seq_len-1 (inclusive)
            # Shape: (seq_len-1, batch, 3, 32, 32)
            targets = batch[:, 1:].transpose(0, 1)
            
            # Forward pass
            preds, z0, z_t = model(x_in, t)
            
            # Loss: Mean Squared Error on reconstructed frames
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # Sauvegarde des poids
    torch.save(model.state_dict(), 'physics_wm_weights.pth')
    print("Modèle sauvegardé sous physics_wm_weights.pth")

if __name__ == '__main__':
    train()
