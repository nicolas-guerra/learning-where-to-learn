import torch
import math
import cupy as cp
from cupyx.scipy.sparse.linalg import gmres, LinearOperator
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["CUDA_PATH"] = "/usr/local/cuda-11.8"
os.environ["PATH"] = "/usr/local/cuda-11.8/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
import pykeops
# pykeops.clean_pykeops()

class MaternGP:
    def __init__(self, size, alpha=2, tau=3, sigma=None, device=None):
        self.device = device
        self.tau = tau
        self.alpha = alpha
        
        if sigma is None:
            sigma = tau**(0.5*(2*alpha - 1))
        self.sigma = sigma
        
        # Use size//2 modes to have sin + cos for each frequency
        n_modes = size // 2
        i = torch.arange(start=1, end=n_modes+1, step=1, device=device)
        self.basis_idx = i

        # Each frequency i has same eigenvalue for sin and cos
        self.sqrt_eig = sigma*((i**2 + tau**2)**(-alpha/2.0))
        
        self.size = size
        
        x = torch.linspace(0, 2*math.pi, size+1, device=device)[:-1] 
        self.grid = x
        
        # Basis: both sin and cos
        norm = math.sqrt(1/math.pi)
        self.phi_sin = torch.sin(i[:, None] * x[None, :]) * norm  # (n_modes, size)
        self.phi_cos = torch.cos(i[:, None] * x[None, :]) * norm  # (n_modes, size)
        
        # Stack: [sin(x), sin(2x), ..., cos(x), cos(2x), ...]
        self.phi = torch.cat([self.phi_sin, self.phi_cos], dim=0)  # (2*n_modes, size)
        
        self.m = torch.zeros(size, device=device)
        
        # Each frequency contributes twice (sin + cos), same eigenvalue
        lam = torch.cat([self.sqrt_eig**2, self.sqrt_eig**2])  # (2*n_modes,)
        self.C = self.phi.T @ torch.diag(lam) @ self.phi
        
    def sample(self, N, z=None):
        if z is None:
            z = torch.randn(N, 2*(self.size//2), device=self.device)  # (N, 2*n_modes)
        
        # Duplicate sqrt_eig for sin and cos
        sqrt_eig_full = torch.cat([self.sqrt_eig, self.sqrt_eig])
        coeff = sqrt_eig_full * z
        u = coeff @ self.phi
        
        return u

class TestDistribution:
    def __init__(self, a, u, device, L=2*math.pi):
        self.a = a
        self.u = u
        self.device = device
        self.N = a.shape[0]
        self.n = a.shape[1]
        self.dx = L/self.n
        self.m = a.mean(0)  # Mean estimate
        self.m2 = torch.mean(self.norm(a)**2)  # Second moment estimate
        
        # Basis on [0, L)
        n_modes = self.n // 2
        i = torch.arange(start=1, end=n_modes+1, step=1, device=device, dtype=a.dtype)
        self.basis_idx = i
        
        x = torch.linspace(0, L, self.n+1, device=device, dtype=a.dtype)[:-1]
        self.grid = x
        
        # Basis Functions: sin and cos
        norm = math.sqrt(1/math.pi)
        phi_sin = torch.sin(i[:, None] * x[None, :]) * norm
        phi_cos = torch.cos(i[:, None] * x[None, :]) * norm
        self.phi = torch.cat([phi_sin, phi_cos], dim=0)  # (2*n_modes, n)
        
        # Project and estimate eigenvalues
        coeff = (a @ self.phi.T) * self.dx  # (N, 2*n_modes)
        self.lam = torch.var(coeff, dim=0, unbiased=False)  # (2*n_modes,) - STORED!
        
        # Reconstruct C in physical space
        self.C = self.phi.T @ torch.diag(self.lam) @ self.phi
    
    def norm(self, a):
        return torch.sqrt(torch.sum(a**2, dim=-1) * self.dx)

class Burgers:
    def __init__(self,N,n,device,L=2*math.pi, nu=1e-1, tmax=1, fudge1=0.9*10, fudge2=0.9*10, stepmax=2e3):
        self.N = N # Number of training samples to generate
        self.n = n # Number of spatial discretization points
        self.device = device
        self.L = L  # Grid is on [0,L]_{per}
        self.nu = nu
        self.tmax = tmax
        self.fudge1 = fudge1
        self.fudge2 = fudge2
        self.stepmax = stepmax
        self.dx = L/n
        self.trunk = torch.linspace(0, L, n+1, device=device)[:-1].unsqueeze(-1) 
        
    def burgers(self, IC):
        """
        Function to solve periodic viscous Burgers' equation in one space dimension.
        Fast evaluation of basic IC to u(1) solution map for Burgers' equation on [0,L]_{per}
        """
        # Derived
        K = IC.shape[-1]
        if K % 2 != 0:
            print("ERROR: ``size'' must be even.")
        k_max = K//2
        k = (2*math.pi/self.L)*torch.arange(start=0, end=k_max + 1, step=1, device=IC.device, dtype=IC.dtype)
        
        # Form mesh in time with CFL constraint (max number of steps is 2000)
        Nt = min(int(self.stepmax), int(self.tmax//min(self.fudge1*self.nu/(torch.abs(IC).max())**2, self.fudge2*(2.79*(self.L/K)**2)/(4*self.nu)))) + 1
        dt = self.tmax/(Nt - 1)
        
        # Precomputed arrays
        g = -1.j*dt*k/2
        g[k_max] = 0 # odd derivative, set K//2 mode to zero
        Emh = torch.exp(-self.nu*(k**2)*dt)
        Emh2 = torch.sqrt(Emh)
            
        # RK4 Integrating Factor
        uhat = torch.fft.rfft(IC)
        for tstep in range(Nt - 1):
            a = g*torch.fft.rfft(torch.fft.irfft(uhat, n=K)**2)
            b = g*torch.fft.rfft(torch.fft.irfft(Emh2*(uhat + a/2), n=K)**2)
            c = g*torch.fft.rfft(torch.fft.irfft(Emh2*uhat + b/2, n=K)**2)
            uhat = Emh*uhat + 1/6*(Emh*a + 2*Emh2*(b + c) + g*torch.fft.rfft(torch.fft.irfft(Emh*uhat + Emh2*c, n=K)**2))
        
        return torch.fft.irfft(uhat, n=K)        

    def train_model(self, a, u):
        N_samples = a.shape[0]
        
        # Train/test split - use 90/10 since we have limited data
        indices = torch.randperm(N_samples)
        ntrain = int(0.9 * N_samples) 
        a_train, a_test = a[indices[:ntrain]], a[indices[ntrain:]]
        u_train, u_test = u[indices[:ntrain]], u[indices[ntrain:]]
            
        # DeepXDE data object
        data = dde.data.TripleCartesianProd(
            X_train=(a_train, self.trunk),
            y_train=u_train,
            X_test=(a_test, self.trunk),
            y_test=u_test,
        )
        
        # Smaller architecture to reduce overfitting
        p = 128  
        net = dde.nn.pytorch.DeepONetCartesianProd(
            layer_sizes_branch=[self.n, 1024, 512, 256, 128, p],
            layer_sizes_trunk=[1, 128, 128, 128, 128, p],
            activation="relu",
            kernel_initializer="Glorot normal"
        )
        net.to(self.device)
        
        # Compile model with learning rate decay
        model = dde.Model(data, net)
        model.compile(
            "adam",
            lr=1e-3,
            loss="mean l2 relative error",
            decay=("inverse time", 2000, 0.9),
        )
        
        # Add L2 regularization via weight decay
        for param_group in model.opt.param_groups:
            param_group['weight_decay'] = 1e-3
        
        print('STARTING MODEL TRAINING')
        
        # Train with early stopping callback - monitor loss_test
        checker = dde.callbacks.EarlyStopping(
            min_delta=1e-3,
            patience=1000,
            baseline=None,
            monitor='loss_test' 
        )
        
        losshistory, train_state = model.train(
            iterations=10000,
            display_every=1000,
            callbacks=[checker],
        )
        
        print(f"\nFinal train loss: {train_state.best_loss_train:.2e}")
        print(f"Final test loss: {train_state.best_loss_test:.2e}")
        
        return model, losshistory, train_state

    def norm(self, a):
        '''
        This function computes the L2 norm of each of the N periodic functions.
        Each function is given as a discretized vector with grid spacing dx.
        a is size [N,n]
        The output is a vector of size N.
        '''
        return torch.sqrt(torch.sum(a**2, dim=-1) * self.dx)

    def ip(self, a, b):
        '''
        This function computes the L2 innerproduct of each of the N pairs of periodic functions.
        Each function is given as a discretized vector with grid spacing dx.
        a and b are both of size [N,n]
        The output is a vector of size N.
        '''
        # Integrate over x (last dimension, columns)
        return torch.sum(a * b, dim=-1) * self.dx  # Shape: (N,)

    def matrix_sqrt(self, C, threshold=1e-6):
        """Matrix square root with pseudoinverse for small eigenvalues"""
        eigvals, eigvecs = torch.linalg.eigh(C)
        
        # Only use significant eigenvalues
        mask = eigvals > threshold
        eigvals_sqrt = torch.zeros_like(eigvals)
        eigvals_sqrt[mask] = torch.sqrt(eigvals[mask])
        
        return eigvecs @ torch.diag(eigvals_sqrt) @ eigvecs.T
    
    def matrix_inv_sqrt(self, C, threshold=1e-6):
        """Matrix inverse square root with pseudoinverse for small eigenvalues"""
        eigvals, eigvecs = torch.linalg.eigh(C)
        
        # Only invert significant eigenvalues
        mask = eigvals > threshold
        eigvals_inv_sqrt = torch.zeros_like(eigvals)
        eigvals_inv_sqrt[mask] = 1.0 / torch.sqrt(eigvals[mask])
        
        return eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T

    def get_W2_sq(self, GP_train, GP_test):
        """
        Compute W2^2 directly in spectral space
        """
        # Mean difference term
        mean_diff_sq = torch.sum((GP_train.m - GP_test.m)**2) * self.dx
        
        # GP_train eigenvalues (duplicate for sin+cos)
        lam_train = torch.cat([GP_train.sqrt_eig**2, GP_train.sqrt_eig**2])
        
        # GP_test eigenvalues (already stored)
        lam_test = GP_test.lam
        
        # Spectral term: sum_i (sqrt(lam_i) - sqrt(lam'_i))^2
        spectral_term = torch.sum((torch.sqrt(lam_train) - torch.sqrt(lam_test))**2)
        
        W2_sq = mean_diff_sq + spectral_term
        
        return W2_sq

    def F(self, alpha, tau, sigma, model, GP_test, z=None):
        '''
        Compute the objective function evaluated at a particular alpha, tau, sigma.
        The objective function uses the given model. Note that the given model
        is not necessarily trained on data from a GP with the given tau.
        To make optimization deterministic, z can be given to generate the same 
        samples for the same alpha, tau, sigma. 
        GP_test is the empirical test distribution fitted to a Gaussian.
        '''     
        Lip = 1.6
        R = 1.6

        # Generate new data from given tau
        GP_train = MaternGP(self.n, alpha=alpha, tau=tau, sigma=sigma, device=self.device)
        a_train = GP_train.sample(self.N, z=z)
        u_train = self.burgers(a_train)
        
        with torch.no_grad():
            term1 = torch.mean(self.norm(u_train - model.net((a_train,self.trunk)))**2)
        
        A = 4*(Lip+R)**4
        with torch.no_grad():
            Ghat_zero = self.norm(model.net((torch.zeros(1,self.n,device=self.device), self.trunk)))
        B = (Lip+R)**2*16*(Ghat_zero**2)
        D = B + A*GP_test.m2
        lam_train_full = torch.cat([GP_train.sqrt_eig**2, GP_train.sqrt_eig**2])
        m2train = torch.sum(lam_train_full)
        term2 = torch.sqrt(A*m2train+D)*torch.sqrt(self.get_W2_sq(GP_train, GP_test))
        return term1+term2

    def grad_F(self, alpha, tau, sigma, model, GP_test, z=None):
        '''
        Gradient of F evaluated at alpha, tau, sigma
        '''
        Lip = 1.6
        R = 1.6
    
        # Generate new data from given parameters
        GP_train = MaternGP(self.n, alpha=alpha, tau=tau, sigma=sigma, device=self.device)
        a_train = GP_train.sample(self.N, z=z)
        u_train = self.burgers(a_train)
    
        # Check scores first - this is often where things go wrong
        scores = self.score(a_train, GP_train)
    
        # TERM 1
        with torch.no_grad():
            errors = self.norm(u_train - model.net((a_train,self.trunk)))**2
        term1 = torch.mean(errors.unsqueeze(1) * scores, dim=0)
    
        # TERM 2
        A = 4*(Lip+R)**4
        with torch.no_grad():
            Ghat_zero = self.norm(model.net((torch.zeros(1,self.n,device=self.device), self.trunk)))
        B = (Lip+R)**2*16*(Ghat_zero**2)
        D = B + A*GP_test.m2
        lam_train_full = torch.cat([GP_train.sqrt_eig**2, GP_train.sqrt_eig**2])
        m2train = torch.sum(lam_train_full)
        W22 = self.get_W2_sq(GP_train, GP_test)
        
        factor = torch.sqrt(W22/(A*m2train+D))
        
        norms_sq = self.norm(a_train)**2
        
        term2_integral = torch.mean((norms_sq/2).unsqueeze(1) * scores, dim=0) 
        term2 = A/2*factor*term2_integral
    
        # TERM 3 & 4 - These involve matrix operations
        C_train = GP_train.C 
        C_test = GP_test.C 
        C_train_sqrt = self.matrix_sqrt(C_train)
        C_train_inv_sqrt = self.matrix_inv_sqrt(C_train)
        middle = C_train_sqrt @ C_test @ C_train_sqrt
        middle_sqrt = self.matrix_sqrt(middle)
        A_prime = C_train_inv_sqrt @ middle_sqrt @ C_train_inv_sqrt
        I = torch.eye(A_prime.shape[0], device=self.device)
        inner_prods = self.ip(a_train, a_train @ (I-A_prime).T) 
        term3_integral = torch.mean((inner_prods/2).unsqueeze(1) * scores, dim=0)
        term3 = term3_integral/factor

        # TERM 4
        m_test = GP_test.m.repeat(self.N,1)
        term4_integral = torch.mean((self.ip(a_train, -m_test)).unsqueeze(1) * scores, dim=0)
        term4 = term4_integral/factor

        return term1 + term2 + term3 + term4 # Shape: (3,)

    def score(self, a_train, GP_train, threshold=1e-6):
        '''Score function'''
        alpha = GP_train.alpha  
        tau = GP_train.tau
        sigma = GP_train.sigma
        lam = GP_train.sqrt_eig**2
        sig_modes = lam > threshold
        
        # Duplicate mask for both sin and cos
        sig_modes_full = torch.cat([sig_modes, sig_modes])  # (2*n_modes,)
        
        phi_sig = GP_train.phi[sig_modes_full]  # Filter both sin and cos
        
        # Duplicate eigenvalues for sin and cos
        lam_full = torch.cat([lam, lam])
        lam_sig = lam_full[sig_modes_full]
        
        # i_sq only depends on frequency, same for sin and cos
        i_sq = GP_train.basis_idx**2
        i_sq_full = torch.cat([i_sq, i_sq])
        i_sq_sig = i_sq_full[sig_modes_full]
        
        coeff = (a_train @ phi_sig.T) * self.dx
        common = (coeff**2 - lam_sig) / (2*lam_sig**2)
        
        dlam_dalpha = -lam_sig * torch.log(i_sq_sig + tau**2)
        dlam_dtau = -2 * alpha * tau * lam_sig / (i_sq_sig + tau**2)
        dlam_dsigma = 2 * lam_sig / sigma

        score_alpha = torch.sum(common * dlam_dalpha, dim=1)
        score_tau = torch.sum(common * dlam_dtau, dim=1)
        score_sigma = torch.sum(common * dlam_dsigma, dim=1)
        
        score = torch.stack([score_alpha, score_tau, score_sigma], dim=1)  # (N, 3)
        
        return score

    def relativeOODerror(self,model,GP_valid):
        num = torch.mean(self.norm(GP_valid.u - model.net((GP_valid.a,self.trunk)))**2)
        den = torch.mean(self.norm(GP_valid.u)**2)
        return torch.sqrt(num/den)
    
        