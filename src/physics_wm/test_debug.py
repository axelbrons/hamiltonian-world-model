import torch
import torch.nn as nn
from src.physics_wm.hnn import HNN, HNN_ODE
from src.physics_wm.encoder_decoder import Encoder, Decoder
from src.physics_wm.model import PhysicsWorldModel
from torchdiffeq import odeint

def test_debug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Debugging sur {device} ---")
    
    latent_dim = 8
    img_size = 32
    in_channels = 9 # 3 frames
    out_channels = 3
    
    print("\n[1] Test Encoder (9 canaux)...")
    encoder = Encoder(in_channels, latent_dim).to(device)
    dummy_input = torch.randn(1, in_channels, img_size, img_size).to(device)
    z0 = encoder(dummy_input)
    print(f"z0 shape: {z0.shape} (Attendu: [1, 16])")
    
    print("\n[2] Test ODE Solver...")
    hnn = HNN(input_dim=latent_dim * 2).to(device)
    ode_func = HNN_ODE(hnn)
    t = torch.linspace(0, 0.4, 3).to(device)
    z_t = odeint(ode_func, z0, t, method='rk4')
    print(f"z_t shape: {z_t.shape}")
    
    print("\n[3] Test Decoder...")
    decoder = Decoder(latent_dim, out_channels).to(device)
    img_out = decoder(z_t[0])
    print(f"Image out shape: {img_out.shape}")
    
    print("\n[4] Test Full Model Forward...")
    model = PhysicsWorldModel(in_channels, out_channels, latent_dim).to(device)
    preds, z0_out, z_t_out = model(dummy_input, t)
    print(f"Forward success, preds: {preds.shape}")

if __name__ == "__main__":
    test_debug()
