import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
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
    dataset_path = f'simple_pendulum_dataset_seq{seq_len}_num1500_diverse_energy.pt'
    if os.path.exists(dataset_path):
        print(f"Loading cached simple pendulum dataset from {dataset_path}...")
        dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    else:
        print("Generating simple pendulum dataset (this may take a minute)...")
        dataset = SimplePendulumDataset(num_sequences=1500, seq_len=seq_len, img_size=32)
        torch.save(dataset, dataset_path)
        print(f"Dataset cached at {dataset_path}")
        
    # Split into 80% Train / 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))
    
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
    
    train_losses = []
    val_losses = []
    lrs = []
    
    print("Starting training with Train/Validation monitoring...")
    for epoch in range(num_epochs):
        # 1. Training Phase
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Random temporal slicing: window of length 20
            window_len = 20
            t0 = np.random.randint(0, seq_len - window_len + 1)
            batch_slice = batch[:, t0 : t0 + window_len]
            
            x_in = torch.cat([batch_slice[:, 0], batch_slice[:, 1], batch_slice[:, 2]], dim=1)
            targets = batch_slice[:, 2:].transpose(0, 1)
            t_slice = t[: window_len - 2]
            
            preds, z0, z_t = model(x_in, t_slice, sub_steps=2)
            total_loss = criterion(preds, targets)
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 2. Validation Phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                window_len = 20
                batch_slice = batch[:, :window_len]
                
                x_in = torch.cat([batch_slice[:, 0], batch_slice[:, 1], batch_slice[:, 2]], dim=1)
                targets = batch_slice[:, 2:].transpose(0, 1)
                t_slice = t[: window_len - 2]
                
                preds, z0, z_t = model(x_in, t_slice, sub_steps=2)
                val_loss = criterion(preds, targets)
                epoch_val_loss += val_loss.item()
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{num_epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

    # Save model weights
    torch.save(model.state_dict(), 'simple_pendulum_weights.pth')
    print("Model saved to simple_pendulum_weights.pth.")

    # 3. Plot Learning Curves
    plt.figure(figsize=(12, 4.5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Train Loss', lw=1.8)
    plt.plot(range(1, num_epochs + 1), val_losses, 'r--', label='Val Loss', lw=1.8)
    plt.xlabel('Époque')
    plt.ylabel('Loss (BCE With Logits)')
    plt.title('Courbes d\'Apprentissage (Train vs Val)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), lrs, 'g-', label='Learning Rate', lw=1.8)
    plt.xlabel('Époque')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Décroissance du Learning Rate (Cosine Annealing)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simple_pendulum_learning_curves.png')
    print("Learning curves saved to simple_pendulum_learning_curves.png.")

if __name__ == '__main__':
    train()
