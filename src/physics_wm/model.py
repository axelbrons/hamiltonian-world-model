import torch
import torch.nn as nn
from torchdiffeq import odeint
from .encoder_decoder import Encoder, Decoder
from .hnn import HNN, HNN_ODE

class PhysicsWorldModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        """
        in_channels: C * seq_len (e.g. 3 * 2 = 6 for 2 RGB frames)
        out_channels: C (e.g. 3 for 1 RGB frame)
        latent_dim: dimension of q (and p)
        """
        super(PhysicsWorldModel, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)
        
        self.hnn = HNN(input_dim=latent_dim * 2)
        self.ode_func = HNN_ODE(self.hnn)
        
    def encode(self, x):
        # x: (batch, in_channels, H, W)
        return self.encoder(x)
        
    def decode(self, z):
        # z: (batch, latent_dim * 2)
        return self.decoder(z)
        
    def forward(self, x, t):
        # 1. Encode to initial state z0 = [q0, p0]
        z0 = self.encode(x)
        
        # 2. Integrate using Neural ODE
        # z_t has shape (len(t), batch, 2 * latent_dim)
        z_t = odeint(self.ode_func, z0, t, method='rk4', options={'step_size': 0.1})
        
        # 3. Decode each timestep
        seq_len, batch_size, latent_dim_2 = z_t.shape
        z_t_flat = z_t.view(-1, latent_dim_2)
        
        preds = self.decode(z_t_flat)
        preds = preds.view(seq_len, batch_size, *preds.shape[1:])
        
        return preds, z0, z_t
