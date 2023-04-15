import torch
from torch.autograd import grad 
import numpy as np
from torchviz import make_dot

def gradients(u, x, allow_unused=True):
    """Compute u_x
    """
    return grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=allow_unused)[0]
    


def mse_loss(y, y_hat):
    return torch.mean((y - y_hat) ** 2)

def ns_pde_loss(x, y, t, u, v, p, nu_t):
    
    rho = 5550.0
    mu = 7.22e-4
    nu = mu / rho
    
    u_t = gradients(u, t)
    v_t = gradients(v, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)
    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)

    continuity = u_x + v_y
    eq_1 = (u_t + (u*u_x + v * u_y) + p_x / rho - (nu + nu_t) * (u_xx + u_yy))
    eq_2 = (v_t + (u*v_x + v * v_y) + p_y / rho - (nu + nu_t) * (v_xx + v_yy))
    # mse for pde loss
    loss_1 = mse_loss(continuity, torch.zeros_like(continuity))
    loss_2 = mse_loss(eq_1, torch.zeros_like(eq_1))
    loss_3 = mse_loss(eq_2, torch.zeros_like(eq_2))
    return [loss_1, loss_2, loss_3]
    