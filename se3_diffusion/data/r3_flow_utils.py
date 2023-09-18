import torch
import numpy as np
beta_min = 0.1
beta_max = 20.0
num_t = 1000
ts = torch.linspace(1e-3, 1.0, num_t)

def alpha_t(t):
    return np.exp(-0.5*(t*beta_min + 0.5*t**2*(beta_max - beta_min)))

def alpha_t_prime(t):
    return -0.5*(beta_min + t*(beta_max - beta_min))*alpha_t(t)

def mu_t(x_0, t):
    return alpha_t(t) * x_0

def sigma_t(t):
    return np.sqrt(1 - alpha_t(t)**2)

def p_t(x_0, t):
    return torch.randn(
        loc=mu_t(x_0, t),
        scale=sigma_t(t)
    )

def vf_t(x, x_0, t):
    alpha_ratio = alpha_t_prime(t) / (1 - alpha_t(t)**2)
    return alpha_ratio * (alpha_t(t) * x - x_0)
