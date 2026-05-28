# Physics-Constrained World Model for Double Pendulum Dynamics

## Overview

This project implements a latent-variable world model that learns the dynamics of a double pendulum system (Acrobot-v1) directly from raw pixel observations. The model combines three architectural components: a convolutional encoder-decoder for visual perception, a Hamiltonian Neural Network (HNN) for learning the conserved energy structure of the underlying physics, and a Neural ODE integrator for continuous-time trajectory rollouts.

The central hypothesis is that physical dynamics admit a symplectic structure that can be exploited as an inductive bias: the true latent state evolves under Hamiltonian mechanics, and the learned representation should respect this constraint. By embedding an HNN within a Neural ODE and training with physics-motivated regularization, the model aims to discover a latent coordinate system (generalised positions q and momenta p) in which the dynamics are exactly Hamiltonian.

## Architecture

### Encoder

A convolutional neural network maps a stack of three consecutive RGB frames (dimensions 3 x 3 x 32 x 32) to a latent vector of dimension `2 * latent_dim`. The encoder consists of four strided convolutional layers (32, 64, 128, 256 channels, stride 2) with LeakyReLU activations, followed by a flatten layer and a linear projection. The output is interpreted as the initial state `z0 = [q0, p0]` in phase space.

### Hamiltonian Neural Network (HNN)

The HNN learns the scalar Hamiltonian (total energy) function `H(q, p)` of the latent dynamics. It is implemented as a fully-connected network with two hidden layers of width 200 and tanh activations, mapping `R^{2d} -> R^1`. The ODE dynamics are derived from the Hamiltonian via Hamilton's equations:

```
dq/dt = dH/dp
dp/dt = -dH/dq
```

The gradients `dH/dq` and `dH/dp` are obtained through automatic differentiation (torch.autograd.grad), ensuring that the learned flow field is exactly symplectic by construction. This formulation guarantees energy conservation and phase-space volume preservation at the function-approximation level, in contrast to unconstrained neural network dynamics.

### Neural ODE Integration

Given the initial latent state `z0` and a time grid `t = [t0, t1, ..., tN]`, the model integrates the HNN-derived ODE using an Euler scheme (step size 0.1) via the torchdiffeq library. The integration produces a trajectory `z(t)` in latent phase space.

### Decoder

A transposed-convolutional network (four upsampling blocks, 256 -> 128 -> 64 -> 32 -> out_channels) maps any latent state `z(t)` from the trajectory back to a single RGB frame. The decoder is symmetric to the encoder and uses LeakyReLU activations throughout.

### Full Model (PhysicsWorldModel)

The forward pass proceeds as:

1. Encode initial observation (three frames) to `z0 = [q0, p0]`.
2. Integrate the HNN ODE over the prescribed time grid to obtain `z_t`.
3. Decode each timestep of `z_t` to obtain predicted image frames.

During training, the model outputs logits; a sigmoid is applied at inference time for visualisation.

## Dataset

The dataset is generated procedurally from the Gymnasium Acrobot-v1 environment. Each episode initialises the double pendulum at the inverted equilibrium (theta_1 = pi, theta_2 = 0, omega_1 = 0, omega_2 = 0) and applies a small random torque to break symmetry, producing a falling trajectory. Frames are captured at 32 x 32 resolution, inverted, thresholded (values below 0.2 set to 0), and normalised to [0, 1]. The dataset stores sequences of length `seq_len` (default 10).

## Training Objective

The loss function comprises three terms:

### Reconstruction Loss

A binary cross-entropy with logits loss (BCEWithLogitsLoss) with a positive class weight of 5.0 to handle sparsity in the thresholded images. The model predicts frames 3 through N (the ground-truth tail of the sequence) conditioned on the first three frames.

### Coordinate Consistency Loss

Hamilton's equations require that `dq/dt = dH/dp`. In the learned coordinate system, this implies a consistency between the time-difference of the position coordinate `q_t` and the momentum coordinate `p_t`:

```
L_cc = mean( || p_{t-1} - (q_t - q_{t-1}) / dt ||^2 )
```

This loss enforces that `p` indeed behaves as the time-derivative of `q`, aligning the learned latent variables with the canonical interpretation of generalised positions and momenta.

### Energy Conservation Loss

For a Hamiltonian system, `H(q(t), p(t))` is a conserved quantity along trajectories. The energy loss penalises fluctuations in the predicted Hamiltonian:

```
L_energy = mean( (H_{t+1} - H_t)^2 )
```

This regularisation encourages the HNN to learn a function that is approximately invariant along the model's own rollouts.

### Full Objective

```
L_total = L_recon + lambda_phys * (L_cc + L_energy)
```

where `lambda_phys` is set to 0.1 after an initial warm-up period of 5 epochs to allow the reconstruction loss to first establish a meaningful latent representation.

## Training Procedure

- Optimiser: Adam, learning rate 1e-3
- Epochs: 60
- Batch size: 16
- Number of training sequences: 300
- Sequence length: 10 frames
- Latent dimension: 8 (16-dimensional phase space)
- Latent time step (dt): 0.2

The model is trained on a CUDA-enabled GPU if available, otherwise falls back to CPU.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports: HNN, HNN_ODE, Encoder, Decoder, PhysicsWorldModel, DoublePendulumDataset |
| `dataset.py` | Procedural dataset generation from Acrobot-v1 Gymnasium environment |
| `encoder_decoder.py` | Convolutional encoder and transposed-convolutional decoder |
| `hnn.py` | Hamiltonian Neural Network and its associated ODE function |
| `model.py` | Full PhysicsWorldModel integrating encoder, HNN-ODE, and decoder |
| `train_wm.py` | Training loop with reconstruction, coordinate-consistency, and energy losses |
| `visualize_wm.py` | Inference and matplotlib visualisation of predicted vs. ground-truth frames |
| `test_debug.py` | Unit tests and shape checks for each component |

## References

- Greydanus, S., Dzamba, M., & Yosinski, J. (2019). *Hamiltonian Neural Networks*. NeurIPS.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS.
- Brockman, G., et al. (2016). *OpenAI Gym*. arXiv:1606.01540.
