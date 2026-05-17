import torch
import torch.nn as nn

class Controller(nn.Module):
    """
    Modèle C (Contrôleur) : Une simple couche linéaire.
    Son rôle est de mapper la vision (z) et la mémoire (h) vers une action (a).
    """
    def __init__(self, z_dim=32, hidden_dim=256, action_dim=3):
        super(Controller, self).__init__()
        
        # Pourquoi un modèle linéaire ?
        # Dans l'article original, David Ha explique que si V et M sont bien entraînés,
        # le Controller peut être très simple. C'est le principe de la séparation 
        # entre la compréhension du monde et la prise de décision.
        
        # Entrée : concaténation de z_t (32) et h_t (256) = 288 dimensions
        input_dim = z_dim + hidden_dim
        self.fc = nn.Linear(input_dim, action_dim)
        
    def forward(self, z, h):
        """
        z : Vecteur latent (batch, z_dim)
        h : État caché du RNN (batch, hidden_dim)
        """
        # On concatène la vision et la mémoire
        # x shape: (batch, 288)
        x = torch.cat([z, h], dim=-1)
        
        # Calcul des actions brutes
        out = self.fc(x)
        
        # Pour l'environnement CarRacing-v3, les actions sont :
        # 1. Steering (Direction) : -1.0 à 1.0 -> Tanh
        # 2. Gas (Accélération) : 0.0 à 1.0 -> Sigmoid
        # 3. Brake (Frein) : 0.0 à 1.0 -> Sigmoid
        
        steering = torch.tanh(out[:, 0])
        gas = torch.sigmoid(out[:, 1])
        brake = torch.sigmoid(out[:, 2])
        
        # On regroupe les actions : (batch, 3)
        return torch.stack([steering, gas, brake], dim=-1)

if __name__ == "__main__":
    # Test rapide des dimensions
    z_dim = 32
    hidden_dim = 256
    model = Controller(z_dim=z_dim, hidden_dim=hidden_dim)
    
    # Simulation d'un batch de 1 état
    dummy_z = torch.randn(1, z_dim)
    dummy_h = torch.randn(1, hidden_dim)
    
    action = model(dummy_z, dummy_h)
    
    print(f"Action prédite (Direction, Gaz, Frein) :\n{action}")
    print(f"Shape de l'action : {action.shape}") # Devrait être (1, 3)
    
    # Vérification des bornes
    print("\nVérification des bornes :")
    print(f"Direction (min/max théorique -1/1) : {action[0,0].item():.4f}")
    print(f"Gaz (min/max théorique 0/1)       : {action[0,1].item():.4f}")
    print(f"Frein (min/max théorique 0/1)     : {action[0,2].item():.4f}")
