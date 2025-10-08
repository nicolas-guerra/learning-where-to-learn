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

class rkhs_func(object):
    def __init__(self,
                 d,
                 J=1000,
                 crange=4,
                 brange=1,
                 sigma=0.25,
                 kernel=gaussian_kernel
                 ):
        super().__init__()
        
        self.d = d
        self.J = J
        self.crange = crange
        self.brange = brange
        self.sigma = sigma
        self.kernel_name = kernel
        
        # Setup
        self.kernel = lambda x,y: self.kernel_name(x, y, self.sigma)
        self.coeff = None
        self.centers = None
        self.set_params()
        
    def set_params(self):
        self.coeff = 2*self.brange*torch.rand(self.J) - self.brange # Unif(-brange,brange)
        self.centers = 2*self.crange*torch.rand(self.J, self.d) - self.crange # Unif(-crange,crange)

    def forward(self, x):
        """
        x: (nbatch, d) tensor, where d is the input dimension.
        Returns (nbatch,) tensor of scalar predictions.
        """
        dev = x.device
        x = self.kernel(x, self.centers.to(dev))
        return torch.einsum('ij,j->i', x, self.coeff.to(dev))
        
    def __call__(self, x):
        return self.forward(x)
