import torch
import torch.nn as nn
from src.physics_wm.hnn import HNN, HNN_ODE
from src.physics_wm.encoder_decoder import Encoder, Decoder
from src.physics_wm.model import PhysicsWorldModel
from torchdiffeq import odeint

def test_debug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Debugging sur {device} ---")
    
    latent_dim = 4
    img_size = 64
    in_channels = 6
    out_channels = 3
    
    # --- 1. TEST ENCODER ---
    print("\n[1] Test Encoder...")
    encoder = Encoder(in_channels, latent_dim).to(device)
    dummy_input = torch.randn(1, in_channels, img_size, img_size).to(device)
    z0 = encoder(dummy_input)
    print(f"z0 shape: {z0.shape} (Attendu: [1, {latent_dim * 2}])")
    print(f"z0 values (mean): {z0.mean().item():.6f}, (std): {z0.std().item():.6f}")
    
    if torch.isnan(z0).any():
        print("!!! ALERTE: z0 contient des NaNs")
    
    # --- 2. TEST SOLVER ODE INDÉPENDANT ---
    print("\n[2] Test ODE Solver...")
    hnn = HNN(input_dim=latent_dim * 2).to(device)
    ode_func = HNN_ODE(hnn)
    t = torch.linspace(0, 0.5, 10).to(device)
    
    # On teste le solveur avec z0
    try:
        # z_t: [len(t), batch, 2*latent_dim]
        z_t = odeint(ode_func, z0, t, method='rk4')
        print(f"z_t shape: {z_t.shape} (Attendu: [10, 1, {latent_dim * 2}])")
        
        # Vérifier si z change au cours du temps
        diff = (z_t[-1] - z_t[0]).abs().sum().item()
        print(f"Évolution totale dans l'espace latent (z_T - z_0): {diff:.8f}")
        
        if diff == 0:
            print("!!! ALERTE: Le vecteur latent ne bouge pas. Le HNN renvoie peut-être un gradient nul.")
    except Exception as e:
        print(f"!!! ERREUR dans le solveur: {e}")

    # --- 3. TEST DECODER ---
    print("\n[3] Test Decoder...")
    decoder = Decoder(latent_dim, out_channels).to(device)
    # On décode le premier et le dernier point de la trajectoire
    with torch.no_grad():
        img_0 = decoder(z_t[0])
        img_T = decoder(z_t[-1])
    
    print(f"Image shape: {img_0.shape}")
    print(f"Image 0 - min: {img_0.min().item():.4f}, max: {img_0.max().item():.4f}, mean: {img_0.mean().item():.4f}")
    print(f"Image T - min: {img_T.min().item():.4f}, max: {img_T.max().item():.4f}, mean: {img_T.mean().item():.4f}")
    
    if img_0.max() == img_0.min():
        print("!!! ALERTE: L'image décodeur est uniforme (probablement toute noire ou toute blanche).")

    # --- 4. TEST INTEGRITÉ DU MODÈLE COMPLET ---
    print("\n[4] Test Full Model Forward...")
    model = PhysicsWorldModel(in_channels, out_channels, latent_dim).to(device)
    try:
        preds, z0_out, z_t_out = model(dummy_input, t)
        print(f"Preds shape: {preds.shape} (Attendu: [10, 1, 3, 32, 32])")
        print("Succès du forward pass complet.")
    except Exception as e:
        print(f"!!! ERREUR dans le modèle complet: {e}")

if __name__ == "__main__":
    test_debug()
