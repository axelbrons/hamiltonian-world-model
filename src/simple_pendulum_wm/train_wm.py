import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import SimplePendulumDataset
from .model import PhysicsWorldModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters for simple pendulum
    num_epochs = 40
    batch_size = 16
    learning_rate = 1e-3
    seq_len = 10
    latent_dim = 1 # 1 for q, 1 for p = 2D phase space
    dt = 0.2
    
    print("Generating simple pendulum dataset...")
    dataset = SimplePendulumDataset(num_sequences=1500, seq_len=seq_len, img_size=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PhysicsWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Weighted BCE to handle background sparsity
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    
    t = torch.linspace(0., (seq_len - 3) * dt, seq_len - 2).to(device)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Physics loss scheduling (warm up for 5 epochs)
        lambda_phys = 0.1 if epoch > 5 else 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Input is the first 3 frames stacked
            x_in = torch.cat([batch[:, 0], batch[:, 1], batch[:, 2]], dim=1)
            # Targets are frames 2 to seq_len-1
            targets = batch[:, 2:].transpose(0, 1)
            
            # Preds shape: (T_pred, B, 3, 32, 32)
            preds, z0, z_t = model(x_in, t)
            
            recon_loss = criterion(preds, targets)
            
            # Coordinate Consistency Loss
            q_t = z_t[:, :, :latent_dim]
            p_t = z_t[:, :, latent_dim:]
            dq = q_t[1:] - q_t[:-1]
            loss_cc = torch.mean((p_t[:-1] - (dq / dt))**2)
            
            # Energy Loss
            H_t = model.hnn(z_t)
            loss_energy = torch.mean(torch.diff(H_t, dim=0)**2)
            
            total_loss = recon_loss + lambda_phys * (loss_cc + loss_energy)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'simple_pendulum_weights.pth')
    print("Model saved to simple_pendulum_weights.pth.")

if __name__ == '__main__':
    train()
