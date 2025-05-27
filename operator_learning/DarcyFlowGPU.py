# # Darcy Flow with GPU
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import pandas as pd
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

class SolveDarcyFlow(nn.Module):
    def __init__(self, M, device=None):
        """
        Solves PDE -div(a*grad(u))=f with DBC on grid MxM including boundaries
        """
        super().__init__()
        self.M = M
        self.h = 1/(M-1)
        self.device = device
    
    def getAx(self, a):
        """
        Build the diagonal matrix of horizontal harmonic means.
        a: (M,M)
        returns Ax: (M*(M-1), M*(M-1)) dense torch.Tensor
        """
        # sums on each horizontal edge
        Ax_den = a[:, :-1] + a[:, 1:] # (M, M−1)
        Ax_matrix = torch.where( 
            Ax_den == 0,                        # if
            torch.zeros_like(Ax_den),           # then
            2 * a[:, :-1] * a[:, 1:] / Ax_den   # else
        )
        Ax_flat = Ax_matrix.reshape(-1) # (M*(M-1),)
        return torch.diag(Ax_flat) # diagonal matrix
    
    def getAy(self, a):
        """
        Build the diagonal matrix of vertical harmonic means.
        a: (M,M)
        returns Ay: (M*(M-1), M*(M-1))
        """
        Ay_den = a[:-1, :] + a[1:, :]                 # (M−1, M)
        Ay_matrix = torch.where(
            Ay_den == 0,
            torch.zeros_like(Ay_den),
            2 * a[:-1, :] * a[1:, :] / Ay_den
        )
        Ay_flat = Ay_matrix.reshape(-1)
        return torch.diag(Ay_flat)
    
    def assemble_operator(self, a):
        """
        Given a (M,M) tensor of conductivity,
        assemble the (M^2 × M^2) finite-difference operator L
        with Dirichlet BC built in.
        """
        M = self.M
        h = self.h

        # 1) edge‐flux diagonals
        Ax = self.getAx(a)                        # (M*(M-1)) x (M*(M-1))
        Ay = self.getAy(a)

        # 2) 1D derivative matrix D: (M−1, M)
        idx = torch.arange(M - 1, device=self.device)
        D = torch.zeros((M - 1, M), device=self.device)
        D[idx, idx]     = -1.0
        D[idx, idx + 1] = +1.0
        D = D / h

        I = torch.eye(M, device=self.device)

        # 3) Kron products to lift to 2D
        Dx = torch.kron(I, D)                     # ((M*M−M) x M^2)
        Dy = torch.kron(D, I)                     # (M^2 x (M*M−M))

        # 4) discrete operator L = Dx.T Ax Dx + Dy.T Ay Dy
        L = Dx.T @ Ax @ Dx + Dy.T @ Ay @ Dy      # (M^2, M^2)

        # 5) enforce Dirichlet BC: row i for any boundary node i has L[i,i]=1, all else 0
        # boundary indices:
        # Row major flattening: idx=r*M+c 
        top    = torch.arange(0, M, device=self.device)            # row 0, all columns from 0 to M-1 
        bottom = torch.arange((M-1)*M, M*M, device=self.device)    # row M-1, all columns from 0 to M-1
        left   = torch.arange(0, M*M, M, device=self.device)       # all rows from 0 to M-1, column 0
        right  = torch.arange(M-1, M*M, M, device=self.device)     # all rows from 0 to M-1, column M-1
        bidx   = torch.unique(torch.cat((top, bottom, left, right)))

        # zero out those rows, then set coef to 1
        L[bidx, :] = 0.0
        L[bidx, bidx] = 1.0

        return L
    
    def forward(self, a, f=None):
        """
        Args:
          a: (N, M, M) tensor of N separate MxM conductivities
          f: (N, M, M) right‐hand side defaults to 1.0 in interior, 0 on boundary

        Returns:
          u: (N, M, M) solutions
        """
        N, M, _ = a.shape

        # default source term
        if f is None:
            f = torch.ones_like(a, device=self.device)
            # zero out boundary
            f[:, 0, :] = 0
            f[:, -1, :] = 0
            f[:, :, 0] = 0
            f[:, :, -1] = 0

        u = torch.zeros_like(a, device=self.device)

        # assemble once per sample
        for n in range(N):
            L = self.assemble_operator(a[n])
            f_flat = f[n].reshape(-1)            # (M^2,)
            # dense solve
            u_flat = torch.linalg.solve(L, f_flat)
            u[n]   = u_flat.reshape(M, M)

        return u

def train_model(M,N_samples,m,alpha,tau,num_epochs=5000,z=None,device=None):
    print('STARTING DATA GENERATION FOR TRAINING')
    # Instantiate GP sampler & PDE solver
    gp = MaternGP(size=M, dim=2, alpha=alpha, tau=tau, device=device)
    solver = SolveDarcyFlow(M=M, device=device)

    # Generate (a,u) pairs
    a = torch.exp(m+gp.sample(N_samples,z=z))     # shape (N_samples, M, M)
    u = solver.forward(a)     # shape (N_samples, M, M)

    # Flatten a and u to input into DeepONet
    a_flat = a.reshape(N_samples, -1)  # (N, M*M)
    u_flat = u.reshape(N_samples, -1)  # (N, M*M)

    # Train/test split
    ntrain = int(0.8 * N_samples)
    a_train, a_test = a_flat[:ntrain], a_flat[ntrain:]
    u_train, u_test = u_flat[:ntrain], u_flat[ntrain:]

    # Trunk grid (same for train & test)
    X, Y = gp.grid
    x_trunk = torch.stack((X.ravel(), Y.ravel())).T  # (M*M, 2)

    # DeepXDE data object
    data = dde.data.TripleCartesianProd(
        X_train=(a_train, x_trunk),
        y_train=u_train,
        X_test=(a_test,  x_trunk),
        y_test=u_test,
    )

    # DeepONet Architecture
    p = 200 # # of DeepONet basis functions
    net = dde.nn.pytorch.DeepONetCartesianProd(
        layer_sizes_branch=[M*M, 256, 256, 128, p],
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

# Objective Function
class Objective:
    def __init__(self, M, model, test_dist, L_0_norm, L_op, Lip, device):
        self.M = M
        self.model = model
        self.test_alpha = test_dist.alpha
        self.test_tau = test_dist.tau
        self.test_sigma = test_dist.sigma
        self.test_m = test_dist.m
        self.test_m2 = test_dist.m2
        self.test_sqrt_eig = test_dist.sqrt_eig
        self.L_0_norm = L_0_norm
        self.L_op = L_op
        self.Lip = Lip
        self.device = device
        
    def part1(self,N_samples,alpha,tau,sigma=None,z=None):
        # Instantiate GP sampler & PDE solver
        gp = MaternGP(size=self.M, dim=2, alpha=alpha, tau=tau,sigma=sigma,device=self.device)
        solver = SolveDarcyFlow(M=self.M, device=self.device)
        if z is None:
            z = torch.randn(N_samples,*gp.size, device=self.device)

        # Generate (a,u) pairs
        a = torch.exp(gp.sample(N_samples,z=z))     # shape (N_samples, M, M)
        u = solver.forward(a)     # shape (N_samples, M, M)

        # Flatten a and u to input into DeepONet
        a_flat = a.reshape(N_samples, -1)  # (N, M*M)
        u_flat = u.reshape(N_samples, -1)  # (N, M*M)
        
        # Trunk grid (same for train & test)
        X, Y = gp.grid
        x_trunk = torch.stack((X.ravel(), Y.ravel())).T  # (M*M, 2)

        # Use DeepONet to predict u
        with torch.no_grad():
            u_flat_pred = self.model.net((a_flat,x_trunk))

        # Return average squared 2-norm
        return torch.mean(torch.norm(u_flat - u_flat_pred, p=2, dim=1)**2)
    
    def part2(self,m,alpha,tau,sigma=None):
        if sigma == None:
            sigma = tau**(alpha - 1)
        i = torch.arange(1, self.M+1, device=self.device)
        I, J = torch.meshgrid(i, i, indexing='ij')
        m2 = torch.norm(m,p=2)**2+torch.sum((sigma**2/(((I*math.pi)**2+(J*math.pi)**2+tau**2)**alpha)))
        return torch.sqrt(torch.mean((self.Lip+self.L_op)**2*(3*(self.Lip+self.L_op)**2*(m2+self.test_m2)+12*self.L_0_norm**2)))
    
    def part3(self,m,alpha,tau,sigma=None):
        if sigma == None:
            sigma = tau**(alpha - 1)
        i = torch.arange(1, self.M+1, device=self.device)
        I, J = torch.meshgrid(i, i, indexing='ij')
        sqrt_eig = sigma/(((I*math.pi)**2+(J*math.pi)**2+tau**2)**(alpha/2)) # Matrix with \sqrt{\lambda_{i,j}}
        return torch.norm(m-self.test_m,p=2)**2+torch.sqrt(torch.mean(torch.sum(torch.sum((sqrt_eig - self.test_sqrt_eig)**2,dim=1),dim=1)))
        
    def value(self,N_samples,m,alpha,tau,sigma=None,z=None):
        """
        Takes the number of samples used to approximate Part 1
        """
        return self.part1(N_samples,alpha,tau,sigma,z=z)+self.part2(m,alpha,tau,sigma=sigma)*self.part3(m,alpha,tau,sigma=sigma)

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
    def __init__(self,darcy_solver,m,alpha,tau,sigma=None,device=None):
        self.darcy_solver = darcy_solver
        self.alpha = alpha
        self.tau = tau
        self.m = m
        if sigma is None:
            self.sigma = self.tau**(self.alpha - 1)
        else:
            self.sigma = sigma
        self.device = device
        self.m2, self.sqrt_eig = self.getStats()
            
    def getStats(self):
        i = torch.arange(1, self.darcy_solver.M+1, device=self.device)
        I, J = torch.meshgrid(i, i, indexing='ij')
        m2 = torch.norm(self.m,p=2)+torch.sum(torch.sum(self.sigma[:,None,None]**2/(((I[None,:,:]*math.pi)**2+(J[None,:,:]*math.pi)**2+self.tau[:,None,None]**2)**self.alpha[:,None,None]),dim=0),dim=0)
        sqrt_eig = self.sigma[:,None,None]/(((I[None,:,:]*math.pi)**2+(J[None,:,:]*math.pi)**2+self.tau[:,None,None]**2)**(self.alpha[:,None,None]/2))
        return m2, sqrt_eig
        
    def computeRelativeOODError(self,model,N_samples,z=None):
        K = len(self.alpha)
        error = 0
        den = 0
        for k in range(K):
            # Instantiate GP sampler
            gp = MaternGP(size=self.darcy_solver.M, dim=2, alpha=self.alpha[k], tau=self.tau[k],sigma=self.sigma[k],device=self.device)
            if z is None:
                z = torch.randn(N_samples,*gp.size, device=self.device)

            # Generate (a,u) pairs
            a = torch.exp(self.m[k]+gp.sample(N_samples,z=z))     # shape (N_samples, M, M)
            u = self.darcy_solver.forward(a)     # shape (N_samples, M, M)

            # Flatten a and u to input into DeepONet
            a_flat = a.reshape(N_samples, -1)  # (N, M*M)
            u_flat = u.reshape(N_samples, -1)  # (N, M*M)

            # Trunk grid (same for train & test)
            X, Y = gp.grid
            x_trunk = torch.stack((X.ravel(), Y.ravel())).T  # (M*M, 2)

            # Use DeepONet to predict u
            with torch.no_grad():
                u_flat_pred = model.net((a_flat,x_trunk))

            # Return average squared 2-norm
            error += torch.mean(torch.norm(u_flat - u_flat_pred, p=2, dim=1)**2)
            den += torch.mean(torch.norm(u_flat, p=2, dim=1)**2)
        return (error/K)/(den/K)

def optimize(init_m, alpha, tau, obj_fun, num_steps,N_samples=1,z=None):
    print('OPTIMIZING GIVEN MODEL')
    if z is None:
        torch.randn(N_samples,obj_fun.M,obj_fun.M,device=obj_fun.device)
    m = torch.tensor(init_m, device=device, requires_grad=True)
    opt = torch.optim.Adam([m], lr=1e-3)
    for step in range(num_steps):
        opt.zero_grad()
        loss = obj_fun.value(N_samples=N_samples,m=m,alpha=alpha,tau=tau,z=z)
        loss.backward()                                 # backprop into m
        opt.step()                                      # update m

        if step % 100 == 0:
            print(f"step {step:03d} | m = {m.item():.4f} | loss = {loss.item():.4e}")
            
    return m.item(), loss.item()

# # Several independent runs of AMA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using Device: {device}')

M=64
darcy_solver = SolveDarcyFlow(M,device=device)

# Hyperparameters
num_AMA_iter = 10
N_samples = 200
N_samples_min = 3
num_train_epochs = 5000
num_min_steps = 500
N_samples_OOD = 3
alpha, tau = 2, 3
K=20 # Number of test dist.
Lip=.5
L_op=.5
L_0_norm=2
runs = 13

trial_results = []
for run in range(runs):
    # Test Distributions
    test_alpha = sample_from_custom_pdf(K,2,3,device=device)
    test_tau = sample_from_custom_pdf(K,1,4,device=device)
    test_m = sample_from_custom_pdf(K,0,1,device=device)
    test_dist = TestDistribution(darcy_solver,test_m,test_alpha,test_tau,device=device)

    # Fix randomness within each run
    z_min = torch.randn(N_samples_min,M,M, device=device)
    z_OOD = torch.randn(N_samples_OOD,M,M, device=device)
    
    m=0.0 # initial m
    AMAloss = 1e+99 # initialize loss
    results = []
    for i in range(num_AMA_iter):
        # Training
        model,_,_ = train_model(M,N_samples,m,alpha,tau,num_epochs=num_train_epochs,device=device)

        # Evaluate Objective at half step
        obj_fun = Objective(M, model, test_dist, L_0_norm, L_op, Lip, device)
        m = torch.tensor(m,device=device)
        temp_AMAloss = obj_fun.value(N_samples=N_samples_min,m=m,alpha=alpha,tau=tau,z=z_min)
        attempt = 0
        while temp_AMAloss > AMAloss and attempt < 2:
            print(f"Hmm, loss {temp_AMAloss} is not smaller than previous loss {AMAloss}. We'll keep training.")
            # TRAIN AGAIN
            losshistory, train_state = model.train(
                 iterations=num_train_epochs,
                 display_every=1000,
                 disregard_previous_best=True,
             )
            temp_AMAloss = obj_fun.value(N_samples=N_samples_min,m=m,alpha=alpha,tau=tau,z=z_min)
            attempt += 1
        AMAloss = temp_AMAloss
        # Evaluate OOD performace    
        OODerror = test_dist.computeRelativeOODError(model,N_samples_OOD,z=z_OOD)
        print(f'Iteration: {i+0.5} | AMA Loss: {AMAloss} | Relative OOD Error: {OODerror} | m: {m}')
        results.append([i+0.5,AMAloss,OODerror,m])
        df = pd.DataFrame(results, columns = ['Iteration','AMA Loss','Relative OOD Error','m'])

        # Minimizing
        temp_m, temp_AMAloss = optimize(m, alpha, tau, obj_fun, num_min_steps,N_samples=N_samples_min,z=z_min)
        attempt = 0
        while temp_AMAloss > AMAloss and attempt < 2:
            temp_m, temp_AMAloss = optimize(temp_m, alpha, tau, obj_fun, num_min_steps,N_samples=N_samples_min,z=z_min)
            attempt += 1
        m, AMAloss = temp_m, temp_AMAloss
        print(f'Iteration: {i+1} | AMA Loss: {AMAloss} | Relative OOD Error: NA | m: {m}')
        results.append([i+1,AMAloss,torch.nan,m])
        df = pd.DataFrame(results, columns = ['Iteration','AMA Loss','Relative OOD Error','m'])
    trial_results.append(df)
    with open('DarcyFlow_results.pkl', 'wb') as f:
        pickle.dump(trial_results, f)