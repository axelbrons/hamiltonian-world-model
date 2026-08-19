import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import SimplePendulumDataset
from .model import PhysicsWorldModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters for simple pendulum (4D phase space)
    num_epochs = 150
    batch_size = 64
    learning_rate = 1e-3 # Standard learning rate for training from scratch
    seq_len = 50
    latent_dim = 2 # 2 for q, 2 for p = 4D phase space (smooth S1 embedding in R2)
    dt = 0.2
    
    # Cache dataset to avoid slow procedural generation on every run
    import os
    dataset_path = f'simple_pendulum_dataset_seq{seq_len}_num1500_diverse_energy.pt'
    if os.path.exists(dataset_path):
        print(f"Loading cached simple pendulum dataset from {dataset_path}...")
        dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    else:
        print("Generating simple pendulum dataset (this may take a minute)...")
        dataset = SimplePendulumDataset(num_sequences=1500, seq_len=seq_len, img_size=32)
        torch.save(dataset, dataset_path)
        print(f"Dataset cached at {dataset_path}")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    
    model = PhysicsWorldModel(in_channels=9, out_channels=3, latent_dim=latent_dim).to(device)
    
    # Resume from checkpoint if it exists
    weights_path = 'simple_pendulum_weights.pth'
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"Loaded pretrained weights from {weights_path} to resume training.")
        except Exception as e:
            print(f"Starting training from scratch ({e})")
            
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Weighted BCE to handle background sparsity
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    
    t = torch.linspace(0., (seq_len - 3) * dt, seq_len - 2).to(device)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Random temporal slicing: choose a random window of length 20
            # (3 context frames + 17 prediction targets) from the 50-frame sequence.
            window_len = 20
            import numpy as np
            t0 = np.random.randint(0, seq_len - window_len + 1)
            batch_slice = batch[:, t0 : t0 + window_len]
            
            # Input is the first 3 frames of the slice stacked
            x_in = torch.cat([batch_slice[:, 0], batch_slice[:, 1], batch_slice[:, 2]], dim=1)
            # Targets are frames 2 to window_len-1 of the slice
            targets = batch_slice[:, 2:].transpose(0, 1)
            
            # Integrate over the corresponding slice time grid
            t_slice = t[: window_len - 2]
            
            # Preds shape: (T_pred, B, 3, 32, 32)
            preds, z0, z_t = model(x_in, t_slice, sub_steps=2)
            
            # Simple reconstruction loss (BCE with logits) as in standard Neural ODE / HNN papers.
            total_loss = criterion(preds, targets)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, LR: {current_lr:.6f}")

    torch.save(model.state_dict(), 'simple_pendulum_weights.pth')
    print("Model saved to simple_pendulum_weights.pth.")

if __name__ == '__main__':
    train()
