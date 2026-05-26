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
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        
        # Décodeur
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

def loss_function(x_reconst, x, mu, log_var):
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconst_loss + kl_div

if __name__ == "__main__":
    input_size = 64*64*3
    model = VAE(input_dim=input_size)
    dummy_input = torch.rand(5, 3, 64, 64)
    reconst, mu, log_var = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {reconst.shape}")
    loss = loss_function(reconst, dummy_input, mu, log_var)
    print(f"Loss value: {loss.item()}")
