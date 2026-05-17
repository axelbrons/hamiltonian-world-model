import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MDNRNN(nn.Module):
    """
    Modèle M (Mémoire) : MDN-RNN (Mixture Density Network - RNN).
    Ce modèle prédit la distribution du prochain état latent z_{t+1} 
    en fonction de l'état actuel z_t et de l'action a_t.
    """
    def __init__(self, z_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5):
        super(MDNRNN, self).__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        
        # 1. Le RNN (LSTM)
        # Entrée : concaténation de z_t (32) et action_t (3) = 35
        self.lstm = nn.LSTM(input_size=z_dim + action_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 2. La couche MDN (Mixture Density Network)
        # Elle doit sortir les paramètres de 'n_gaussians' distributions normales pour chaque dimension de z.
        # Pour chaque gaussienne, on a besoin de :
        # - pi : le poids (probabilité) de la gaussienne (taille n_gaussians)
        # - mu : la moyenne pour chaque dimension de z (taille n_gaussians * z_dim)
        # - sigma : l'écart-type pour chaque dimension de z (taille n_gaussians * z_dim)
        
        n_out = n_gaussians * (1 + 2 * z_dim)
        self.fc_mdn = nn.Linear(hidden_dim, n_out)
        
    def forward(self, z, action, hidden_state=None):
        """
        z : (batch, seq_len, z_dim)
        action : (batch, seq_len, action_dim)
        """
        # Concaténer l'état latent et l'action
        # x shape: (batch, seq_len, 35)
        x = torch.cat([z, action], dim=-1)
        
        # Passage dans le LSTM
        # out shape: (batch, seq_len, hidden_dim)
        out, hidden_state = self.lstm(x, hidden_state)
        
        # Passage dans la couche MDN
        y = self.fc_mdn(out)
        
        # Découpage des sorties
        # pi (poids) : on veut que la somme soit égale à 1
        pi = y[..., :self.n_gaussians]
        pi = F.softmax(pi, dim=-1)
        
        # mu (moyennes) et sigma (écarts-types)
        mu_sigma = y[..., self.n_gaussians:]
        
        # mu: (batch, seq_len, n_gaussians, z_dim)
        mu = mu_sigma[..., :self.n_gaussians * self.z_dim]
        mu = mu.view(mu.size(0), mu.size(1), self.n_gaussians, self.z_dim)
        
        # sigma: (batch, seq_len, n_gaussians, z_dim)
        sigma = mu_sigma[..., self.n_gaussians * self.z_dim:]
        sigma = sigma.view(sigma.size(0), sigma.size(1), self.n_gaussians, self.z_dim)
        # On utilise l'exponentielle pour garantir que sigma soit toujours strictement positif
        sigma = torch.exp(sigma)
        
        return pi, mu, sigma, hidden_state

def mdn_loss_function(out_pi, out_mu, out_sigma, target_z):
    """
    Calcule la log-vraisemblance négative d'un mélange de gaussiennes.
    On veut maximiser la probabilité que target_z vienne de la distribution prédite.
    """
    # target_z shape: (batch, seq_len, z_dim) -> (batch, seq_len, 1, z_dim) pour le broadcast
    target = target_z.unsqueeze(2) 
    
    # Calcul de la probabilité de target sous chaque gaussienne (Loi Normale multidimensionnelle)
    # exponent = -0.5 * sum( ((x - mu) / sigma)^2 )
    exponent = -0.5 * torch.sum(torch.pow((target - out_mu) / out_sigma, 2), dim=3)
    
    # log_prob = exponent - log(prod(sigma * sqrt(2*pi)))
    #          = exponent - sum(log(sigma)) - (z_dim/2) * log(2*pi)
    z_dim = target_z.size(2)
    log_prob = exponent - torch.sum(torch.log(out_sigma), dim=3) - 0.5 * z_dim * math.log(2 * math.pi)
    
    # Probabilité totale = Somme (pi_i * prob_i)
    # En log : log(Somme(exp(log_pi + log_prob))) -> logsumexp
    log_weighted_prob = torch.log(out_pi) + log_prob
    loss = -torch.logsumexp(log_weighted_prob, dim=2)
    
    return torch.mean(loss)

if __name__ == "__main__":
    # Test rapide des dimensions
    z_dim = 32
    action_dim = 3
    model = MDNRNN(z_dim=z_dim, action_dim=action_dim)
    
    # Simulation d'un batch de 2 séquences de longueur 10
    dummy_z = torch.randn(2, 10, z_dim)
    dummy_action = torch.randn(2, 10, action_dim)
    
    pi, mu, sigma, _ = model(dummy_z, dummy_action)
    
    print(f"Pi shape (poids du mélange): {pi.shape}")      # (2, 10, 5)
    print(f"Mu shape (moyennes): {mu.shape}")              # (2, 10, 5, 32)
    print(f"Sigma shape (écarts-types): {sigma.shape}")   # (2, 10, 5, 32)
    
    # Test de la fonction de perte
    target_z = torch.randn(2, 10, z_dim)
    loss = mdn_loss_function(pi, mu, sigma, target_z)
    print(f"MDN Loss: {loss.item():.4f}")
