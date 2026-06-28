import torch
import torch.nn as nn
from torchdiffeq import odeint
from .encoder_decoder import Encoder, Decoder
from .hnn import HNN, HNN_ODE

class PhysicsWorldModel(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        """
        in_channels: C * seq_len (e.g. 3 * 3 = 9 for 3 RGB frames)
        out_channels: C (e.g. 3 for 1 RGB frame)
        latent_dim: dimension of q (and p)
        """
        super(PhysicsWorldModel, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels)
        
        self.hnn = HNN(input_dim=latent_dim * 2)
        # Learnable friction/damping coefficient
        self.gamma = nn.Parameter(torch.tensor([0.05]))
        self.ode_func = HNN_ODE(self.hnn, parent_model=self)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def leapfrog_step(self, z, dt):
        d = z.shape[-1] // 2
        
        with torch.enable_grad():
            z_with_grad = z.detach().requires_grad_(True) if not z.requires_grad else z
            q = z_with_grad[:, :d]
            p = z_with_grad[:, d:]
            
            gamma = torch.abs(self.gamma)
            # Step 1: p_half = (p - (dt / 2) * dH/dq) / (1 + (dt / 2) * gamma)
            z_k = torch.cat([q, p], dim=-1)
            H_k = self.hnn(z_k)
            dH_dq = torch.autograd.grad(H_k.sum(), q, create_graph=True)[0]
            p_half = (p - (dt / 2.0) * dH_dq) / (1.0 + (dt / 2.0) * gamma)
            
            # Step 2: q_next = q + dt * dH/dp
            z_half = torch.cat([q, p_half], dim=-1)
            H_half = self.hnn(z_half)
            dH_dp = torch.autograd.grad(H_half.sum(), p_half, create_graph=True)[0]
            q_next = q + dt * dH_dp
            
            # Step 3: p_next = (p_half - (dt / 2) * dH/dq2) / (1 + (dt / 2) * gamma)
            z_half2 = torch.cat([q_next, p_half], dim=-1)
            H_half2 = self.hnn(z_half2)
            dH_dq2 = torch.autograd.grad(H_half2.sum(), q_next, create_graph=True)[0]
            p_next = (p_half - (dt / 2.0) * dH_dq2) / (1.0 + (dt / 2.0) * gamma)
            
            z_next = torch.cat([q_next, p_next], dim=-1)
            
        if not z.requires_grad:
            z_next = z_next.detach()
            
        return z_next

    def integrate_leapfrog(self, z0, t):
        dt = (t[1] - t[0]).item()
        sub_steps = 4
        sub_dt = dt / sub_steps
        
        z_t = [z0]
        z = z0
        
        for _ in range(len(t) - 1):
            for _ in range(sub_steps):
                z = self.leapfrog_step(z, sub_dt)
            z_t.append(z)
            
        return torch.stack(z_t, dim=0)

    def forward(self, x, t, solver='leapfrog'):
        # 1. Encode to initial state z0 = [q0, p0]
        z0 = self.encode(x)
        
        # 2. Integrate using Selected Solver
        if solver == 'leapfrog':
            z_t = self.integrate_leapfrog(z0, t)
        else:
            z_t = odeint(self.ode_func, z0, t, method='rk4', options={'step_size': 0.05})
        
        if self.training:
            # Latent noise injection (Solution 3)
            z_t = z_t + torch.randn_like(z_t) * 1e-3
        
        # 3. Decode each timestep
        seq_len, batch_size, latent_dim_2 = z_t.shape
        z_t_flat = z_t.view(-1, latent_dim_2)
        
        preds = self.decode(z_t_flat)
        preds = preds.view(seq_len, batch_size, *preds.shape[1:])
        
        return preds, z0, z_t
