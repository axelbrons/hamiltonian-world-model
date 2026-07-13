import torch
import torch.nn as nn

class HNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=200):
        super(HNN, self).__init__()
        # input_dim should be 2*d (q and p)
        self.d = input_dim // 2
        self.net_q = nn.Sequential(
            nn.Linear(self.d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Potential energy V(q)
        )
        # Initialize final layer of net_q to 0.
        # This ensures a flat initial potential energy landscape to prevent chaotic
        # dynamics at start and avoid black image collapse.
        nn.init.zeros_(self.net_q[-1].weight)
        nn.init.zeros_(self.net_q[-1].bias)
    
    def forward(self, x):
        q = x[..., :self.d]
        p = x[..., self.d:]
        # Quadratic kinetic energy T(p) = 0.5 * p^2
        # This guarantees dH/dp = p (meaning velocity dq/dt = p) and prevents gradient explosion.
        V = self.net_q(q)
        T = 0.5 * torch.sum(p**2, dim=-1, keepdim=True)
        return V + T

class HNN_ODE(nn.Module):
    def __init__(self, hnn, parent_model=None):
        super(HNN_ODE, self).__init__()
        self.hnn = hnn
        # Wrap parent_model in a list to prevent PyTorch from registering it as a submodule
        # which would cause a circular dependency and RecursionError on model.to(device).
        self.parent_model_list = [parent_model]
        
    def forward(self, t, x):
        # x is [batch, 2*d]
        with torch.enable_grad():
            x = x.requires_grad_(True)
            H = self.hnn(x)
            
            # dH/dx
            grad_H = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            
        # x = [q, p]
        # dq/dt = dH/dp
        # dp/dt = -dH/dq - gamma * p
        d = x.shape[-1] // 2
        
        dH_dq = grad_H[:, :d]
        dH_dp = grad_H[:, d:]
        
        dq_dt = dH_dp
        
        parent_model = self.parent_model_list[0]
        if parent_model is not None and hasattr(parent_model, 'gamma'):
            p = x[:, d:]
            gamma = torch.abs(parent_model.gamma)
            dp_dt = -dH_dq - gamma * p
        else:
            dp_dt = -dH_dq
        
        return torch.cat([dq_dt, dp_dt], dim=-1)
