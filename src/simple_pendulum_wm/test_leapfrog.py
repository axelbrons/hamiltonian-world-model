import torch
import torch.nn as nn

class SimpleHNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.net(x)

hnn = SimpleHNN()

def leapfrog_step(z, dt):
    d = z.shape[-1] // 2
    
    with torch.enable_grad():
        z_with_grad = z.detach().requires_grad_(True) if not z.requires_grad else z
        q = z_with_grad[:, :d]
        p = z_with_grad[:, d:]
        
        # Step 1
        z_k = torch.cat([q, p], dim=-1)
        H_k = hnn(z_k)
        dH_dq = torch.autograd.grad(H_k.sum(), q, create_graph=True)[0]
        p_half = p - (dt / 2.0) * dH_dq
        
        # Step 2
        z_half = torch.cat([q, p_half], dim=-1)
        H_half = hnn(z_half)
        dH_dp = torch.autograd.grad(H_half.sum(), p_half, create_graph=True)[0]
        q_next = q + dt * dH_dp
        
        # Step 3
        z_half2 = torch.cat([q_next, p_half], dim=-1)
        H_half2 = hnn(z_half2)
        dH_dq2 = torch.autograd.grad(H_half2.sum(), q_next, create_graph=True)[0]
        p_next = p_half - (dt / 2.0) * dH_dq2
        
        z_next = torch.cat([q_next, p_next], dim=-1)
        
    if not z.requires_grad:
        z_next = z_next.detach()
        
    return z_next

# Test 1: Training mode (requires_grad = True)
z0 = torch.randn(4, 2, requires_grad=True)
z = z0
dt = 0.1
for _ in range(5):
    z = leapfrog_step(z, dt)

loss = z.sum()
loss.backward()
print("Test 1 (requires_grad=True) passed. z0 gradient is not None:", z0.grad is not None)

# Test 2: Inference mode (torch.no_grad())
z0_no_grad = torch.randn(4, 2)
with torch.no_grad():
    z_eval = z0_no_grad
    for _ in range(5):
        z_eval = leapfrog_step(z_eval, dt)
    print("Test 2 (no_grad) passed. Output z_eval shape:", z_eval.shape)
