#!/usr/bin/env python
# coding: utf-8

# # AMA NtD with GPU 

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import pickle

class MaternGP:
    """
    Matern Gaussian Process
    If dim=1, then returns samples of size 'size'.
    If dim=2, then returns samples of size 'size x size'
    """
    def __init__(self, size, dim=1, alpha=2, tau=3, sigma=None, device=None):
        self.device = device
        self.dim = dim
        
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))
            
        if self.dim == 1:
            i_max = size
            i = torch.arange(start=1, end=i_max+1, step=1, device=device)
            self.basis_idx = i
    
            self.sqrt_eig = sigma*(((math.pi**2)*(i**2) + tau**2)**(-alpha/2.0))
            
            self.size = (size,)
            
            x = torch.linspace(0, 1, size, device=device) 
            self.grid = x
            # Basis Functions
            self.phi = torch.sin(math.pi * i[:, None] * x[None, :])
        
        if self.dim == 2:
            i_max = size
            j_max = size
            i = torch.arange(start=1, end=i_max+1, step=1, device=device)
            j = torch.arange(start=1, end=j_max+1, step=1, device=device)
            I, J = torch.meshgrid(i,j,indexing='ij')
        
            self.sqrt_eig = sigma*(((math.pi**2)*(I**2) + (math.pi**2)*(J**2) + tau**2)**(-alpha/2.0))
            
            self.size = (size, size)
            
            x = torch.linspace(0, 1, size, device=device)
            y = torch.linspace(0, 1, size, device=device)
            self.grid = torch.meshgrid(x,y,indexing='xy')
            
            # Basis Functions
            self.phi_x = torch.sin(math.pi * i[:, None] * x[None, :])  # (i_max, M)
            self.phi_y = torch.sin(math.pi * j[:, None] * y[None, :])  # (j_max, M)
        
    def sample(self, N, z=None):
        """
        Returns N samples from the Matern GP
        """
        if z is None:
            z = torch.randn(N,*self.size, device=self.device)
        coeff = self.sqrt_eig*z  # if dim==1, N x size. if dim==2, N x size x size
        
        if self.dim == 1:
            u = math.sqrt(2) * (coeff @ self.phi)  # N x size
            
        if self.dim == 2:            
            u = torch.einsum('nij,ik,jl->nkl', coeff, self.phi_x, self.phi_y)
            u = 2 * u  # (N, size, size)
        
        return u

class DtNOperator(nn.Module):
    def __init__(self, a, device=None):
        """
        Solves PDE -div(a*grad(u))=f with DBC on grid MxM including boundaries
        and return NBC
        """
        super().__init__()
        self.a = a
        self.M = self.a.shape[0]
        self.h = 1/(self.M-1)
        self.device = device
        self.Ax = self.getAx()
        self.Ay = self.getAy()
        self.L = self.assemble_operator()
        self.D = self.build_D()
        self.D_pinv = torch.linalg.pinv(self.D)
        self.trunk = self.getTrunk()
    
    def getTrunk(self):
        top_x = torch.linspace(0,1,self.M)
        top_x = top_x[0:-1]
        right_x = torch.ones((self.M-1,))
        bottom_x = torch.linspace(1,0,self.M)
        bottom_x = bottom_x[0:-1]
        left_x = torch.zeros((self.M-1,))
        x = torch.concatenate((top_x,right_x,bottom_x,left_x))
        x = x.reshape(4*self.M-4,1)
        top_y = torch.ones((self.M-1,))
        right_y = torch.linspace(1,0,self.M)
        right_y = right_y[0:-1]
        bottom_y = torch.zeros((self.M-1,))
        left_y = torch.linspace(0,1,self.M)
        left_y = left_y[0:-1]
        y = torch.concatenate((top_y,right_y,bottom_y,left_y))
        y = y.reshape(4*self.M-4,1)
        trunk = torch.concatenate((x,y),1)
        return trunk
    
    def getAx(self):
        """
        Build the diagonal matrix of horizontal harmonic means from self.a
        returns Ax: (M*(M-1), M*(M-1)) dense torch.Tensor
        """
        # sums on each horizontal edge
        Ax_den = self.a[:, :-1] + self.a[:, 1:] # (M, M−1)
        Ax_matrix = torch.where( 
            Ax_den == 0,                        # if
            torch.zeros_like(Ax_den),           # then
            2 * self.a[:, :-1] * self.a[:, 1:] / Ax_den   # else
        )
        Ax_flat = Ax_matrix.reshape(-1) # (M*(M-1),)
        return torch.diag(Ax_flat) # diagonal matrix
    
    def getAy(self):
        """
        Build the diagonal matrix of vertical harmonic means from self.a
        returns Ay: (M*(M-1), M*(M-1))
        """
        Ay_den = self.a[:-1, :] + self.a[1:, :]                 # (M−1, M)
        Ay_matrix = torch.where(
            Ay_den == 0,
            torch.zeros_like(Ay_den),
            2 * self.a[:-1, :] * self.a[1:, :] / Ay_den
        )
        Ay_flat = Ay_matrix.reshape(-1)
        return torch.diag(Ay_flat)
    
    def assemble_operator(self):
        """
        Using (M,M) tensor of conductivity self.a,
        assemble the (M^2 × M^2) finite-difference operator L
        with Dirichlet BC built in.
        """

        # 1D derivative matrix D: (M−1, M)
        idx = torch.arange(self.M - 1, device=self.device)
        D = torch.zeros((self.M - 1, self.M), device=self.device)
        D[idx, idx]     = -1.0
        D[idx, idx + 1] = +1.0
        D = D / self.h

        I = torch.eye(self.M, device=self.device)

        # Kron products to lift to 2D
        Dx = torch.kron(I, D)                     # ((M*M−M) x M^2)
        Dy = torch.kron(D, I)                     # (M^2 x (M*M−M))

        # Discrete operator L = Dx.T Ax Dx + Dy.T Ay Dy
        L = Dx.T @ self.Ax @ Dx + Dy.T @ self.Ay @ Dy      # (M^2, M^2)

        # Enforce Dirichlet BC: row i for any boundary node i has L[i,i]=1, all else 0
        # boundary indices:
        # Row major flattening: idx=r*M+c 
        top    = torch.arange(0, self.M, device=self.device)            # row 0, all columns from 0 to M-1 
        bottom = torch.arange((self.M-1)*self.M, self.M*self.M, device=self.device)    # row M-1, all columns from 0 to M-1
        left   = torch.arange(0, self.M*self.M, self.M, device=self.device)       # all rows from 0 to M-1, column 0
        right  = torch.arange(self.M-1, self.M*self.M, self.M, device=self.device)     # all rows from 0 to M-1, column M-1
        bidx   = torch.unique(torch.cat((top, bottom, left, right)))

        # zero out those rows, then set coef to 1
        L[bidx, :] = 0.0
        L[bidx, bidx] = 1.0

        return L
    
    def neumann_flux(self, u):
        """
        Compute the Neumann flux q = a * du/dn on the boundary of an M×M grid,
        including corners (using the average flux in vertical and horizontal dir.), and return a 1D array
        of length 4*(M-1) going clockwise starting at the top‐left corner.
        """
        assert u.shape == (self.M, self.M) and self.a.shape == (self.M, self.M)
        flux_vals = []

        # Top‐left corner (i=0,j=0)
        dudn_vert = (u[0, 0] - u[1, 0]) / self.h      # normal points upward
        dudn_horz = (u[0, 0] - u[0, 1]) / self.h      # normal points leftward
        flux_vals.append(self.a[0, 0] * 0.5 * (dudn_vert + dudn_horz))

        # Top edge interior (i=0, j=1..M-2)
        for j in range(1, self.M - 1):
            dudn = (u[0, j] - u[1, j]) / self.h
            flux_vals.append(self.a[0, j] * dudn)

        # Top‐right corner (i=0,j=M-1)
        dudn_vert = (u[0, self.M - 1] - u[1, self.M - 1]) / self.h
        dudn_horz = (u[0, self.M - 1] - u[0, self.M - 2]) / self.h
        flux_vals.append(self.a[0, self.M - 1] * 0.5 * (dudn_vert + dudn_horz))

        # Right edge interior (j=M-1, i=1..M-2)
        for i in range(1, self.M - 1):
            dudn = (u[i, self.M - 1] - u[i, self.M - 2]) / self.h
            flux_vals.append(self.a[i, self.M - 1] * dudn)

        # Bottom‐right corner (i=M-1,j=M-1)
        dudn_vert = (u[self.M - 1, self.M - 1] - u[self.M - 2, self.M - 1]) / self.h  # normal points downward
        dudn_horz = (u[self.M - 1, self.M - 1] - u[self.M - 1, self.M - 2]) / self.h  # normal points rightward
        flux_vals.append(self.a[self.M - 1, self.M - 1] * 0.5 * (dudn_vert + dudn_horz))

        # Bottom edge interior (i=M-1, j=M-2..1)
        for j in range(self.M - 2, 0, -1):
            dudn = (u[self.M - 1, j] - u[self.M - 2, j]) / self.h
            flux_vals.append(self.a[self.M - 1, j] * dudn)

        # Bottom‐left corner (i=M-1,j=0)
        dudn_vert = (u[self.M - 1, 0] - u[self.M - 2, 0]) / self.h
        dudn_horz = (u[self.M - 1, 0] - u[self.M - 1, 1]) / self.h
        flux_vals.append(self.a[self.M - 1, 0] * 0.5 * (dudn_vert + dudn_horz))

        # Left edge interior (j=0, i=M-2..1)
        for i in range(self.M - 2, 0, -1):
            dudn = (u[i, 0] - u[i, 1]) / self.h
            flux_vals.append(self.a[i, 0] * dudn)

        return torch.tensor(flux_vals,device=self.device)
    
    def forward(self, dbc, f=None):
        """
        Args:
          dbc: (N, 4*M-4) tensor of N separate Dirichlet boundary conditions
                          of size 4M-4
          f:   (N, M, M) right‐hand side defaults to 0 in interior
        Returns:
          nbc: (N, 4*M-4) tensor of N separate Neumann boundary conditions
                          of size 4M-4
        """
        assert dbc.shape[1] == 4*self.M-4
        N = dbc.shape[0]

        # default source term
        if f is None:
            f = torch.zeros((N,self.M,self.M), device=self.device)

        u = torch.zeros((N,self.M,self.M), device=self.device)
        nbc = torch.zeros_like(dbc, device=self.device)
        # assemble once per sample
        for n in range(N):
            
            # Apply boundary conditions
            f[n,0,0:self.M-1] = dbc[n,:self.M-1]
            f[n,:self.M-1,self.M-1] = dbc[n,self.M-1:2*self.M - 2]
            bottom_values = dbc[n,2*self.M-2:3*self.M-3]
            f[n,self.M-1,1:] = bottom_values.flip(0)
            left_values = dbc[n,3*self.M-3:]
            f[n,1:, 0] = left_values.flip(0)
            f_flat = f[n].reshape(-1)            # (M^2,)
            
            # Dense solve
            u_flat = torch.linalg.solve(self.L, f_flat)
            u[n] = u_flat.reshape(self.M, self.M)
            
            # Compute NBC
            nbc[n] = self.neumann_flux(u[n])
            nbc[n] -= torch.mean(nbc[n]) # Enforce NBC integrates to 0
        return nbc
    
    def build_D(self):
        """
        Build the DtN matrix D of shape (4*M-4, 4*M-4), so that
          nbc = D @ dbc

        M : int, grid size (including boundary)
        a : (M,M) conductivity array
        """
        n_rows = 4*self.M - 4       # number of Neumann outputs
        n_cols = 4*self.M - 4       # length of full DBC
        D = torch.zeros((n_rows, n_cols), device=self.device)

        # basis e_j in the full-DBC space
        print('CONSTRUCTING OPERATOR')
        for j in range(n_cols):
            e = torch.zeros(1, n_cols, device=self.device)
            e[0,j] = 1.0
            # Apply DtN to Basis
            D[:, j] = self.forward(e)[0]
            if j % 10 == 0:
                print(f"Completed column {j+1}/{n_cols}")
        return D

def train_model(DtN,N_samples,omega,alpha,tau,num_epochs=10000,z=None,device=None):
    print('STARTING DATA GENERATION FOR TRAINING')
    M = DtN.M
    D_pinv = DtN.D_pinv
    trunk = DtN.trunk
    # Instantiate NBC GP sampler
    nbc_gp = MaternGP(size=4*M-4, dim=1, alpha=alpha, tau=tau, device=device)

    # Generate (nbc,dbc) pairs
    nbc = amp*torch.sin(omega*(nbc_gp.grid-1/2))+nbc_gp.sample(N_samples,z=z) # Shape (N_samples, 4M-4)
    nbc -= torch.mean(nbc,dim=1).reshape(-1,1) # Enforce mean 0
    dbc = nbc @ D_pinv.T

    # Train/test split
    ntrain = int(0.8 * N_samples)
    nbc_train, nbc_test = nbc[:ntrain], nbc[ntrain:]
    dbc_train, dbc_test = dbc[:ntrain], dbc[ntrain:]

    # DeepXDE data object
    data = dde.data.TripleCartesianProd(
        X_train=(nbc_train, trunk),
        y_train=dbc_train,
        X_test=(nbc_test,  trunk),
        y_test=dbc_test,
    )

    # DeepONet Architecture
    p = 200 # # of DeepONet basis functions
    net = dde.nn.pytorch.DeepONetCartesianProd(
        layer_sizes_branch=[4*M-4, 256, 256, 128, p],
        layer_sizes_trunk=[2, 128, 128, 128, p],
        activation="relu",
        kernel_initializer="Glorot normal",
    )
    net.to(device)  # Ensure the model params live on GPU

    # Compile model
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        loss="mean l2 relative error",
        metrics=[],
    )

    print('STARTING MODEL TRAINING')
    # Train model
    losshistory, train_state = model.train(
        iterations=num_epochs,
        display_every=1000,
    )
    return model, losshistory, train_state

class Objective:
    def __init__(self, DtN, model, test_dist, Lip, Lip_model, L0_norm, L0_norm_model, device):
        self.DtN = DtN
        self.model = model
        self.test_omega = test_dist.omega
        self.test_alpha = test_dist.alpha
        self.test_tau = test_dist.tau
        self.test_sigma = test_dist.sigma
        self.test_m2 = test_dist.m2
        self.test_sqrt_eig = test_dist.sqrt_eig
        self.L0_norm = L0_norm
        self.L0_norm_model = L0_norm_model
        self.Lip = Lip
        self.Lip_model = Lip_model
        self.device = device
        
    def part1(self,N_samples,omega,alpha,tau,sigma=None,z=None):
        M = self.DtN.M
        D_pinv = self.DtN.D_pinv
        trunk = self.DtN.trunk
        # Instantiate NBC GP sampler
        nbc_gp = MaternGP(size=4*M-4, dim=1, alpha=alpha, tau=tau, sigma=sigma, device=device)
        
        # Generate (nbc,dbc) pairs
        nbc = amp*torch.sin(omega*(nbc_gp.grid-1/2))+nbc_gp.sample(N_samples,z=z) # Shape (N_samples, 4M-4)
        nbc -= torch.mean(nbc,dim=1).reshape(-1,1) # Enforce mean 0
        dbc = nbc @ D_pinv.T

        # Use DeepONet to predict u
        with torch.no_grad():
            dbc_pred = self.model.net((nbc,trunk))

        # Return average squared 2-norm
        return torch.mean(torch.norm(dbc - dbc_pred, p=2, dim=1)**2)
    
    def part2(self,omega,alpha,tau,sigma=None):
        if sigma == None:
            sigma = tau**(alpha - 1/2)
        i = torch.arange(1, self.DtN.M+1, device=self.device)
        x = torch.linspace(0,1,4*self.DtN.M-4,device=self.device)
        mean_norm_squared = torch.norm(amp*torch.sin(omega*(x-0.5)),p=2)**2
        m2 = mean_norm_squared+torch.sum((sigma**2/(((i*math.pi)**2+tau**2)**alpha)))
        return torch.sqrt(torch.mean((self.Lip+self.Lip_model)**2*(4*(self.Lip+self.Lip_model)**2*(m2+self.test_m2)+16*(self.L0_norm**2+self.L0_norm_model**2))))
    
    def part3(self,omega,alpha,tau,sigma=None):
        if sigma == None:
            sigma = tau**(alpha - 1/2)
        i = torch.arange(1, self.DtN.M+1, device=self.device)
        sqrt_eig = sigma/(((i*math.pi)**2+tau**2)**(alpha/2)) # vector of \sqrt{\lambda_{i}}
        x = torch.linspace(0,1,4*self.DtN.M-4,device=self.device)
        mean_diff = torch.norm(amp*torch.sin(omega*(x[None,:]-0.5)) - amp*torch.sin(self.test_omega[:,None]*(x[None,:]-0.5)),p=2,dim=1)**2
        return torch.sqrt(torch.mean(mean_diff + torch.sum((sqrt_eig - self.test_sqrt_eig)**2,dim=1)))
        
    def value(self,N_samples,omega,alpha,tau,sigma=None,z=None):
        """
        Takes the number of samples used to approximate Part 1
        """
        return self.part1(N_samples,omega,alpha,tau,sigma=sigma,z=z)+self.part2(omega,alpha,tau,sigma=sigma)*self.part3(omega,alpha,tau,sigma=sigma)

def sample_from_custom_pdf(num_samples, min_x, max_x, device=None, num_grid=10001):
    """
    Sample from the PDF f(x) = C * sin(x-pi/2)+1 on x in [min_x, max_x]
    C is some normalization constant
    using inverse‐CDF sampling with careful unique‐point handling.
    Args:
        num_samples (int): Number of samples to draw.
        num_grid    (int): Number of grid points to approximate CDF (default 10001).
    Returns:
        samples (ndarray): Array of shape (num_samples,) of draws from f.
    """
    # Define a fine grid on [-1, 1]
    x = torch.linspace(min_x, max_x, num_grid, device=device)
    # Compute unnormalized PDF
    pdf = torch.sin(x-math.pi/2)+1
    # Approximate CDF via the trapezoidal rule
    dx = x[1] - x[0]
    cdf = torch.cumsum(pdf,dim=0) * dx
    cdf = cdf / cdf[-1]     # normalize so cdf[-1] == 1
    cdf[0] = 0.0            # enforce CDF(–1) = 0 exactly

    # Remove any repeated values (from zero-density regions)
    mask = torch.ones_like(cdf, dtype=torch.bool)
    mask[1:] = cdf[1:] != cdf[:-1]
    cdf_u = cdf[mask]
    x_u   = x[mask]

    # Draw uniform samples
    u = torch.rand(num_samples, device=device)

    # Invert CDF by linear interpolation
    idx = torch.searchsorted(cdf_u, u, right=True)
    idx = torch.clamp(idx, 1, cdf_u.shape[0] - 1)
    idx0 = idx - 1
    idx1 = idx
    c0 = cdf_u[idx0]
    c1 = cdf_u[idx1]
    x0 = x_u[idx0]
    x1 = x_u[idx1]
    
    # Interpolate
    t = (u - c0) / (c1 - c0)
    samples = x0 + t * (x1 - x0)
    return samples

class TestDistribution:
    def __init__(self,DtN,omega,alpha,tau,sigma=None,device=None):
        self.DtN = DtN
        self.omega = omega
        self.alpha = alpha
        self.tau = tau
        if sigma is None:
            self.sigma = self.tau**(self.alpha - 1/2)
        else:
            self.sigma = sigma
        self.device = device
        self.m2, self.sqrt_eig = self.getStats()
    
    def getStats(self):
        i = torch.arange(1, self.DtN.M+1, device=device)
        x = torch.linspace(0,1,4*self.DtN.M-4,device=device)
        mean_norm_squared = torch.norm(amp*torch.sin(self.omega[:,None]*(x[None,:]-0.5)),p=2,dim=1)**2
        m2 = mean_norm_squared + torch.sum(self.sigma[:,None]**2/(((i[None,:]*math.pi)**2+self.tau[:,None]**2)**self.alpha[:,None]),dim=1)
        sqrt_eig = self.sigma[:,None]/(((i[None,:]*math.pi)**2+self.tau[:,None]**2)**(self.alpha[:,None]/2)) # K x (M+1)
        return m2, sqrt_eig
        
    def computeRelativeOODError(self,model,N_samples,z=None):
        K = len(self.alpha)
        error = 0
        den = 0
        for k in range(K):
            M = self.DtN.M
            D_pinv = self.DtN.D_pinv
            trunk = self.DtN.trunk
            # Instantiate NBC GP sampler
            nbc_gp = MaternGP(size=4*M-4, dim=1, alpha=self.alpha[k], tau=self.tau[k], device=device)

            # Generate (nbc,dbc) pairs
            nbc = amp*torch.sin(self.omega[k]*(nbc_gp.grid-1/2))+nbc_gp.sample(N_samples,z=z) # Shape (N_samples, 4M-4)
            nbc -= torch.mean(nbc,dim=1).reshape(-1,1) # Enforce mean 0
            dbc = nbc @ D_pinv.T

            # Use DeepONet to predict u
            with torch.no_grad():
                dbc_pred = model.net((nbc,trunk))

            # Add average relative error for particular k
            error += torch.mean(torch.norm(dbc - dbc_pred, p=2, dim=1)**2)
            den += torch.mean(torch.norm(dbc, p=2, dim=1)**2)
        return (error/K)/(den/K)

# # Several independent runs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using Device: {device}')

# Conductivity
torch.manual_seed(0)
a_np = np.load('conductivity.npy').astype(np.float32)
a = torch.from_numpy(a_np).to('cuda')
DtN = DtNOperator(a,device=device) 

# Hyperparameters
runs = 5
num_AMA_iter = 10
num_train_epochs = 5000
N_samples_train = 1000
N_samples_min = 1000
N_samples_OOD = 1000

alpha, tau = 2.0,3.0 
Lip = 1.0
Lip_model = 1.0
L0_norm= 0.0
L0_norm_model= 0.1
amp = torch.tensor(1,device=device)

trial_results = []
for run in range(runs):
    # Test Distributions
    K = 20
    # test_alpha = sample_from_custom_pdf(K,2,8,device=device)
    test_alpha = 1.5*torch.ones(K,device=device)
    test_tau = sample_from_custom_pdf(K,0,50,device=device)
    test_omega = sample_from_custom_pdf(K,-4*math.pi,4*math.pi,device=device)
    # test_alpha = 2*torch.ones(K,device=device)
    # test_tau = 3*torch.ones(K,device=device)
    # test_omega = 5*torch.ones(K,device=device)
    test_dist = TestDistribution(DtN,test_omega,test_alpha,test_tau,device=device)
    
    # Fix randomness within each run
    z_min = torch.randn(N_samples_min,4*DtN.M-4, device=device)
    z_OOD = torch.randn(N_samples_OOD,4*DtN.M-4, device=device)
    
    omega = 4*math.pi # initial omega
    AMAloss = 1e+99 # initialize AMA Loss
    results = []
    for i in range(num_AMA_iter):
        model,_,_ = train_model(DtN,N_samples_train,omega,alpha,tau,num_epochs=num_train_epochs,device=device)

        # Evaluate objective function at half step
        obj_fun = Objective(DtN, model, test_dist, Lip, Lip_model, L0_norm, L0_norm_model, device=device)
        omega = torch.tensor(omega,device=device)
        temp_AMAloss = obj_fun.value(N_samples=N_samples_min,omega=omega,alpha=alpha,tau=tau,z=z_min)
        attempt = 0
        while temp_AMAloss > AMAloss and attempt < 10:
            print(f"Hmm, loss {temp_AMAloss} is not smaller than previous loss {AMAloss}. We'll keep training.")
            # TRAIN AGAIN
            losshistory, train_state = model.train(
                 iterations=1000,
                 display_every=100,
                 disregard_previous_best=True,
             )
            temp_AMAloss = obj_fun.value(N_samples=N_samples_min,omega=omega,alpha=alpha,tau=tau,z=z_min)
            attempt += 1
        AMAloss = temp_AMAloss
        
        # Evaluate OOD performance
        OODerror = test_dist.computeRelativeOODError(model,N_samples_OOD,z=z_OOD)
        print(f'Iteration {i+0.5} | AMA Loss {AMAloss} | OOD Error {OODerror} | omega {omega}')
        results.append([i+0.5,AMAloss,OODerror,omega])
        df = pd.DataFrame(results, columns = ['Iteration','AMA Loss','Relative OOD Error','omega'])

        # Minimizing
        print("MINIMIZING")
        tol = 1e-12
        # Wrap the obj_fun for differential_evolution (Output CPU)
        def obj_w(omega):
            return obj_fun.value(N_samples=N_samples_min,
                                 omega=torch.tensor(omega,dtype=torch.float32,device=device),
                                 alpha=alpha,
                                 tau=tau,
                                 z=z_min).item()

        res = differential_evolution(obj_w, bounds=[(-4*np.pi, 4*np.pi)], tol=tol,maxiter=5000, popsize=30)
        temp_omega = res.x[0]
        temp_AMAloss = res.fun
        attempt = 0
        while temp_AMAloss >= AMAloss and attempt < 10:
            print(f"Hmm, loss {temp_AMAloss} is not less than previous loss {AMAloss}. Let's use smaller tolerance for minimizing.")
            tol = tol / 2
            res = differential_evolution(obj_w, bounds=[(-4*np.pi, 4*np.pi)], tol=tol,maxiter=5000, popsize=30)
            temp_omega = res.x[0]
            temp_AMAloss = res.fun
            attempt += 1
        omega = torch.tensor(temp_omega, dtype=torch.float32, device=device)
        AMAloss = torch.tensor(temp_AMAloss, dtype=torch.float32, device=device)
        
        print(f'Iteration {i+1} | AMA Loss {AMAloss} | OOD Error NA | omega {omega}')
        results.append([i+1,AMAloss,torch.nan,omega])
        df = pd.DataFrame(results, columns = ['Iteration','AMA Loss','Relative OOD Error','omega'])
    trial_results.append(df)
    with open('NtD_results.pkl', 'wb') as f:
        pickle.dump(trial_results, f)
