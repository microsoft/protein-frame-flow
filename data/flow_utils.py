import torch

beta_min = 0.1
beta_max = 10.0
num_t = 1000

def alpha_t(t):
    return torch.exp(-0.5*(t*beta_min + 0.5*t**2*(beta_max - beta_min)))

def alpha_t_prime(t):
    return -0.5*(beta_min + t*(beta_max - beta_min))*alpha_t(t)

def mu_t(x_0, t):
    return alpha_t(t) * x_0

def sigma_t(t):
    return torch.sqrt(1 - alpha_t(t)**2)

def reschedule(t):
    return 1 - sigma_t(1-t)
