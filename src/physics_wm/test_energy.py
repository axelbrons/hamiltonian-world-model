import gymnasium as gym
import numpy as np

def acrobot_energy(state):
    theta1, theta2, dtheta1, dtheta2 = state[0], state[1], state[2], state[3]
    m11 = 3.5 + np.cos(theta2)
    m22 = 1.25
    m12 = 1.25 + 0.5 * np.cos(theta2)
    T = 0.5 * (m11 * dtheta1**2 + 2.0 * m12 * dtheta1 * dtheta2 + m22 * dtheta2**2)
    V = - 9.8 * (1.5 * np.cos(theta1) + 0.5 * np.cos(theta1 + theta2))
    return T + V

env = gym.make('Acrobot-v1', render_mode='rgb_array')
env.reset()
env.unwrapped.state = np.array([1.0, -0.5, 0.2, -0.1], dtype=np.float32)

print("Initial state:", env.unwrapped.state)
print("Initial energy:", acrobot_energy(env.unwrapped.state))

for i in range(10):
    env.step(1) # Passive step
    print(f"Step {i+1} state:", env.unwrapped.state)
    print(f"Step {i+1} energy:", acrobot_energy(env.unwrapped.state))
