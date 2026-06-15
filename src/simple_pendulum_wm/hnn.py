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
        self.net_p = nn.Sequential(
            nn.Linear(self.d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Kinetic energy T(p)
        )
    
    def forward(self, x):
        q = x[..., :self.d]
        p = x[..., self.d:]
        return self.net_q(q) + self.net_p(p)

class HNN_ODE(nn.Module):
    def __init__(self, hnn):
        super(HNN_ODE, self).__init__()
        self.hnn = hnn
        
    def forward(self, t, x):
        # x is [batch, 2*d]
        with torch.enable_grad():
            x = x.requires_grad_(True)
            H = self.hnn(x)
            
            # dH/dx
            grad_H = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            
        # x = [q, p]
        # dq/dt = dH/dp
        # dp/dt = -dH/dq
        d = x.shape[-1] // 2
        
        dH_dq = grad_H[:, :d]
        dH_dp = grad_H[:, d:]
        
        dq_dt = dH_dp
        dp_dt = -dH_dq
        
        return torch.cat([dq_dt, dp_dt], dim=-1)
