import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) basique inspiré du notebook MNIST.
    Adapté pour des images de CarRacing (64x64x3).
    """
    def __init__(self, input_dim=64*64*3, h_dim=400, z_dim=32):
        super(VAE, self).__init__()
        
        # Encodeur
        # fc1: Reçoit l'image aplatie (64*64*3 = 12288)
        self.fc1 = nn.Linear(input_dim, h_dim)
        # fc2 & fc3: Produisent la moyenne (mu) et le log de la variance (log_var)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        
        # Décodeur
        # fc4: Reçoit le vecteur latent z (taille z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        # fc5: Reconstruit l'image aplatie
        self.fc5 = nn.Linear(h_dim, input_dim)
        
    def encode(self, x):
        """Transforme l'image en paramètres de la distribution latente (mu, log_var)."""
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        """Astuce de reparamétrisation pour permettre la backpropagation."""
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Reconstruit l'image à partir du vecteur latent z."""
        h = F.relu(self.fc4(z))
        # Sigmoid car les pixels sont normalisés entre 0 et 1
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        """Passage complet : encodage -> reparamétrisation -> décodage."""
        # On aplatit l'image si elle ne l'est pas déjà
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

def loss_function(x_reconst, x, mu, log_var):
    """
    Calcule la perte totale du VAE : Perte de Reconstruction + Divergence KL.
    """
    # 1. Perte de reconstruction (Binary Cross Entropy)
    # On compare l'image reconstruite avec l'image originale aplatie
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    
    reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
    
    # 2. Divergence KL (Régularisation)
    # Force la distribution latente à ressembler à une loi normale centrée réduite N(0, 1)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return reconst_loss + kl_div

if __name__ == "__main__":
    # Petit test pour vérifier que les dimensions fonctionnent
    input_size = 64*64*3
    model = VAE(input_dim=input_size)
    
    # Simulation d'un batch de 5 images CarRacing (déjà aplaties ou non)
    dummy_input = torch.rand(5, 3, 64, 64)
    reconst, mu, log_var = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {reconst.shape}") # Devrait être (5, 12288)
    print(f"Mu shape: {mu.shape}")
    print(f"Log_var shape: {log_var.shape}")
    
    loss = loss_function(reconst, dummy_input, mu, log_var)
    print(f"Loss value: {loss.item()}")
