import torch
from kernels import gaussian_kernel


def sobol_func(x):
    """
    x: (nbatch, d) tensor, where d is the input dimension
    """
    index = torch.arange(1, x.shape[-1] + 1, 1, device=x.device)
    index = (index - 2) / 2
    x = torch.abs(4 * x - 2) + index
    x /= (1 + index)
    x = torch.prod(x, dim=-1)
    return x

def ishigimi_func(x):
    return (1 + 0.1*x[..., 2]**4)*torch.sin(x[..., 0]) + 7*torch.sin(x[..., 1])**2

def friedmann1_func(x):
    out = 10*torch.sin(torch.pi*x[..., 0]*x[..., 1])
    out += 20*(x[..., 2] - 0.5)**2 + 10*x[..., 3] + 5*x[..., 4]
    
    return out

def friedmann2_func(x):
    tmp = 520*torch.pi*x[..., 1] + 40*torch.pi
    out = tmp * ( 1 + 10*x[..., 3])
    out = 1. / out
    out = x[..., 2]*tmp - out
    out = (100*x[..., 0])**2 + out**2

    return torch.sqrt(out)
    
