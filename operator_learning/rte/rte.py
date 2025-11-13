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
import ot
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class MaternGP:
    """
    Matern Gaussian Process
    If dim=1, then returns samples of size 'size'.
    If dim=2, then returns samples of size 'size x size'.
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
    
class RTESolver:
    def __init__(self, L, dr, dtheta, ep_index, device):
        self.L = L
        self.dr = dr
        self.device = device
        self.x = torch.arange(0, L+dr, self.dr, device=device)
        self.dtheta = dtheta
        self.theta0 = torch.arange(-math.pi + dtheta/2, math.pi - dtheta/2 + dtheta, dtheta, device=device)
        self.ep_index = ep_index
        self.epsilon = 1.0 / (2 ** (ep_index))
        self.Nr = len(self.x)
        self.Nt = len(self.theta0)        
        
        # Trunk
        # Create 2D meshgrids
        xx0, yy0 = torch.meshgrid(self.x, self.x, indexing='xy')
        # Reshape coordinates in the same way
        x_vec = xx0.reshape(-1)
        y_vec = yy0.reshape(-1).flipud()
        # Stack into a single (Nr^2, 2) tensor
        self.trunk = torch.stack([x_vec, y_vec], dim=1)
        self.Lip = None
        self.R = None
        self.mean_test_m2 = None
        self.true_model_zero = None
        self.c_test = None
        self.a_test = None
        self.u_test = None
        
        xx, yy, vv = torch.meshgrid(self.x, self.x, self.theta0, indexing='xy')
        self.in_index = torch.zeros(xx.shape,device=self.device)
        self.in_index[1:-1,self.x==0,torch.cos(self.theta0)>0] = 1
        self.in_index[1:-1,torch.isclose(self.x,torch.tensor(self.L,device=device)),torch.cos(self.theta0)<0] = 1
        self.in_index[torch.isclose(self.x,torch.tensor(self.L,device=device)),1:-1,torch.sin(self.theta0)<0] = 1
        self.in_index[self.x==0,1:-1,torch.sin(self.theta0)>0] = 1
        self.in_mask = self.in_index.bool()

        self.out_index = torch.zeros(xx.shape,device=self.device)
        self.out_index[1:-1,self.x==0,torch.cos(self.theta0)<0] = 1
        self.out_index[1:-1,torch.isclose(self.x,torch.tensor(self.L,device=device)),torch.cos(self.theta0)>0] = 1
        self.out_index[torch.isclose(self.x,torch.tensor(self.L,device=device)),1:-1,torch.sin(self.theta0)>0] = 1
        self.out_index[self.x==0,1:-1,torch.sin(self.theta0)<0] = 1
        self.out_mask = self.out_index.bool()

        self.transport_mask = (~self.in_mask) & (~self.out_mask)
    
    def MAB_multi(self,p,a):
        p = p.view(self.Nt, self.Nr, self.Nr).permute(2, 1, 0).contiguous()

        # Pre-compute sin/cos for all angles
        sinT = torch.sin(self.theta0)
        cosT = torch.cos(self.theta0)

        # Compute gradients for all time steps at once
        Rp_pre_x = (p[:,1:,:] - p[:,:-1,:])/self.dr
        Rp_pre_y = (p[1:,:,:] - p[:-1,:,:])/self.dr

        # Initialize output tensors
        Rp_x = torch.zeros_like(p,device=self.device)
        Rp_y = torch.zeros_like(p,device=self.device)
        boundary_D = torch.zeros_like(p,device=self.device)

        # Vectorized y-direction updates
        mask_sin_neg = sinT <= 0
        mask_sin_pos = sinT > 0

        Rp_y[:-1,:,:] = torch.where(mask_sin_neg.view(1,1,-1), 
                                    sinT.view(1,1,-1) * Rp_pre_y, 
                                    torch.zeros_like(Rp_pre_y))
        Rp_y[1:,:,:] += torch.where(mask_sin_pos.view(1,1,-1),
                                    sinT.view(1,1,-1) * Rp_pre_y,
                                    torch.zeros_like(Rp_pre_y))
        Rp_y[:,0,:] = 0
        Rp_y[:,-1,:] = 0

        # Vectorized x-direction updates  
        mask_cos_pos = cosT >= 0
        mask_cos_neg = cosT < 0

        Rp_x[:,1:,:] = torch.where(mask_cos_pos.view(1,1,-1),
                                   cosT.view(1,1,-1) * Rp_pre_x,
                                   torch.zeros_like(Rp_pre_x))
        Rp_x[:,:-1,:] += torch.where(mask_cos_neg.view(1,1,-1),
                                    cosT.view(1,1,-1) * Rp_pre_x,
                                    torch.zeros_like(Rp_pre_x))
        Rp_x[0,:,:] = 0
        Rp_x[-1,:,:] = 0

        # Boundary conditions (vectorized)
        boundary_D[-1,:,:] = torch.where(mask_sin_neg, p[-1,:,:], boundary_D[-1,:,:])
        boundary_D[0,:,:] = torch.where(mask_sin_pos, p[0,:,:], boundary_D[0,:,:])
        boundary_D[:,0,:] = torch.where(mask_cos_pos, p[:,0,:], boundary_D[:,0,:])
        boundary_D[:,-1,:] = torch.where(mask_cos_neg, p[:,-1,:], boundary_D[:,-1,:])

        # BCp calculation
        do_apply = self.transport_mask.any(dim=2)           # (Nr, Nr) boolean       
        mean_p = p.mean(dim=2, keepdim=True)          # (Nr, Nr, 1)
        p_demeaned = p - mean_p
        BCp = a.view(self.Nr, self.Nr, 1) * p_demeaned
        BCp = BCp * do_apply.unsqueeze(-1).to(BCp.dtype)


        Mb = self.epsilon*Rp_x + self.epsilon*Rp_y + BCp + boundary_D
        Mb = Mb.permute(2, 1, 0).contiguous().view(-1)

        return Mb

    def SolveRTE(self,a,verbose=False):
        '''Given scattering coefficient 'a' as a matrix, this function
        solves the RTE and returns the solution 'u' integrated over velocity'''

        # Boundary Condition
        phi3 = torch.zeros(self.Nr, self.Nr, self.Nt, device=self.device)
        g = 1 + 0.5 * torch.sin(math.pi * self.x / self.L)
        for kt in range(self.Nt): 
            ang_factor = torch.clamp(torch.cos(self.theta0[kt]), min=0.0)  # max(0, cos(theta))
            phi3[:, 0, kt] = g * ang_factor  # left boundary only (column 0)

        # RHS
        RHS = torch.zeros(self.Nr, self.Nr, self.Nt, device=self.device)
        RHS[self.in_mask] = phi3[self.in_mask]
        RHS_vec = RHS.permute(2, 1, 0).contiguous().view(-1)

        # Use GMRES to solve Ax=b where Ax operator A is given with matvec_gpu
        # Convert PyTorch tensor to CuPy array
        RHS_vec_cp = cp.asarray(RHS_vec.detach())
        def matvec_gpu(p_cp):
            # Convert CuPy array to torch
            p_torch = torch.as_tensor(p_cp, device=self.device)
            result_torch = self.MAB_multi(p_torch, a)
            # Convert back to CuPy array
            result_cp = cp.asarray(result_torch.detach())
            return result_cp
        N = self.Nr * self.Nr * self.Nt
        A_gpu = LinearOperator((N, N), matvec=matvec_gpu, dtype=cp.float64)
        f_cp, info = gmres(A_gpu, RHS_vec_cp, restart=300, tol=1e-8)
        # Print convergence information
        if verbose:
            if info == 0:
                final_residual = cp.linalg.norm(A_gpu @ f_cp - RHS_vec_cp) / cp.linalg.norm(RHS_vec_cp)
                print(f"GMRES converged to relative residual {final_residual:.1e}")
            elif info > 0:
                print(f"GMRES reached maximum iterations ({info}) without convergence")
            else:
                print(f"GMRES failed with error code {info}")
        # convert result back to torch
        f_torch = torch.as_tensor(f_cp, device=self.device)

        # Convert soln to tensor and integrate over velocity
        u3 = f_torch.view(self.Nt, self.Nr, self.Nr).permute(2, 1, 0).contiguous()  # Reshape into tensor using MATLAB ordering
        u_mat = torch.sum(u3, 2) * self.dtheta

        return u_mat, u3
       
    def MAB_adjoint_multi(self,p,a):
        p = p.view(self.Nt, self.Nr, self.Nr).permute(2, 1, 0).contiguous()

        # Pre-compute sin/cos for all angles
        sinT = torch.sin(self.theta0)
        cosT = torch.cos(self.theta0)

        # Compute gradients for all time steps at once
        Rp_pre_x = (p[:,1:,:] - p[:,:-1,:])/self.dr
        Rp_pre_y = (p[1:,:,:] - p[:-1,:,:])/self.dr

        # Initialize output tensors
        Rp_x = torch.zeros_like(p,device=self.device)
        Rp_y = torch.zeros_like(p,device=self.device)
        boundary_D = torch.zeros_like(p,device=self.device)

        # Vectorized y-direction updates
        mask_sin_neg = sinT <= 0
        mask_sin_pos = sinT > 0

        # Swap instead of e.g. :-1 we save from 1: compared to forward solve. 
        # Apply identity and save to left out dimension. 
        # In this e.g., save to index 0 which is what boundary_D does
        Rp_y[1:,:,:] = torch.where(mask_sin_neg.view(1,1,-1), 
                                    sinT.view(1,1,-1) * Rp_pre_y, 
                                    torch.zeros_like(Rp_pre_y))
        Rp_y[:-1,:,:] += torch.where(mask_sin_pos.view(1,1,-1),
                                    sinT.view(1,1,-1) * Rp_pre_y,
                                    torch.zeros_like(Rp_pre_y))
        Rp_y[:,0,:] = 0
        Rp_y[:,-1,:] = 0

        # Vectorized x-direction updates  
        mask_cos_pos = cosT >= 0
        mask_cos_neg = cosT < 0

        Rp_x[:,:-1,:] = torch.where(mask_cos_pos.view(1,1,-1),
                                   cosT.view(1,1,-1) * Rp_pre_x,
                                   torch.zeros_like(Rp_pre_x))
        Rp_x[:,1:,:] += torch.where(mask_cos_neg.view(1,1,-1),
                                    cosT.view(1,1,-1) * Rp_pre_x,
                                    torch.zeros_like(Rp_pre_x))
        Rp_x[0,:,:] = 0
        Rp_x[-1,:,:] = 0

        # Boundary conditions (vectorized)
        boundary_D[0,:,:] = torch.where(mask_sin_neg, p[0,:,:], boundary_D[0,:,:])
        boundary_D[-1,:,:] = torch.where(mask_sin_pos, p[-1,:,:], boundary_D[-1,:,:])
        boundary_D[:,-1,:] = torch.where(mask_cos_pos, p[:,-1,:], boundary_D[:,-1,:])
        boundary_D[:,0,:] = torch.where(mask_cos_neg, p[:,0,:], boundary_D[:,0,:])

        # BCp calculation
        do_apply = self.transport_mask.any(dim=2)           # (Nr, Nr) boolean       
        mean_p = p.mean(dim=2, keepdim=True)          # (Nr, Nr, 1)
        p_demeaned = mean_p - p                       # Flip compared to forward solve, sign difference
        BCp = a.view(self.Nr, self.Nr, 1) * p_demeaned
        BCp = BCp * do_apply.unsqueeze(-1).to(BCp.dtype)

        Mb = self.epsilon*Rp_x + self.epsilon*Rp_y + BCp + boundary_D
        Mb = Mb.permute(2, 1, 0).contiguous().view(-1)

        return Mb

    def SolveAdjointRTE(self,a,w,verbose=False):
        '''Given scattering coefficient 'a' as a matrix, this function
        solves the RTE and returns the solution 'u' integrated over velocity'''

        # RHS
        RHS = torch.zeros(self.Nr, self.Nr, self.Nt, device=self.device)
        RHS[self.out_mask] = 0 # Outflow boundary condition
        RHS[self.transport_mask] = self.epsilon*w[self.transport_mask]
        RHS_vec = RHS.permute(2, 1, 0).contiguous().view(-1)

        # Use GMRES to solve Ax=b where Ax operator A is given with matvec_gpu
        # Convert PyTorch tensor to CuPy array
        RHS_vec_cp = cp.asarray(RHS_vec.detach())
        def matvec_gpu(p_cp):
            # Convert CuPy array to torch
            p_torch = torch.as_tensor(p_cp, device=self.device)
            result_torch = self.MAB_adjoint_multi(p_torch, a)
            # Convert back to CuPy array
            result_cp = cp.asarray(result_torch.detach())
            return result_cp
        N = self.Nr * self.Nr * self.Nt
        A_gpu = LinearOperator((N, N), matvec=matvec_gpu, dtype=cp.float64)
        # call CuPy GMRES
        f_cp, info = gmres(A_gpu, RHS_vec_cp, restart=300, tol=1e-8)
        # Print convergence information
        if verbose:
            if info == 0:
                final_residual = cp.linalg.norm(A_gpu @ f_cp - RHS_vec_cp) / cp.linalg.norm(RHS_vec_cp)
                print(f"GMRES converged to relative residual {final_residual:.1e}")
            elif info > 0:
                print(f"GMRES reached maximum iterations ({info}) without convergence")
            else:
                print(f"GMRES failed with error code {info}")
        # convert result back to torch
        f_torch = torch.as_tensor(f_cp, device=self.device)

        # Convert soln to tensor
        u3 = f_torch.view(self.Nt, self.Nr, self.Nr).permute(2, 1, 0).contiguous()  # Reshape into tensor using MATLAB ordering

        return u3

    def compute_u(self, a, get_soln=False):
        print('Computing Forward Solutions.')
        N = a.shape[0]
        
        if get_soln:
            u = torch.zeros_like(a, device=self.device)
            f = torch.zeros((N,self.Nr,self.Nr,self.Nt), device=self.device)
            
            # Sequential loop with progress bar
            for i in tqdm(range(N), desc="Processing samples"):
                u[i], f[i] = self.SolveRTE(a[i])

            return u, f
            
        else:
            u = torch.zeros_like(a, device=self.device)

            # Sequential loop with progress bar
            for i in tqdm(range(N), desc="Processing samples"):
                u[i] = self.SolveRTE(a[i])[0]

            return u
        
    def compute_adjoint(self,a_train,w):
        print('Computing Adjoint Solutions.')
        N = a_train.shape[0]
        g = torch.zeros_like(w)
        for i in tqdm(range(N), desc="Processing samples"):
            g[i] = self.SolveAdjointRTE(a_train[i],w[i])
        return g
    
    def compute_solver_gradient(self,f,g):
        return -(1.0 / float(self.epsilon)) * torch.sum((torch.mean(f, dim=3, keepdim=True) - f) * g, dim=3) * self.dtheta
        
    def train_model(self,a,u):
        N_samples = a.shape[0]

        # Train/test split
        ntrain = int(0.8 * N_samples)
        a_vec = a.reshape(N_samples, self.Nr**2) # Reshape images into long vectors
        u_vec = u.reshape(N_samples, self.Nr**2)
        a_train, a_test = a_vec[:ntrain], a_vec[ntrain:]
        u_train, u_test = u_vec[:ntrain], u_vec[ntrain:]

        # DeepXDE data object
        data = dde.data.TripleCartesianProd(
            X_train=(a_train, self.trunk),
            y_train=u_train,
            X_test=(a_test,  self.trunk),
            y_test=u_test,
        )
        
        # Simpler architecture
        p = 32  
        net = dde.nn.pytorch.DeepONetCartesianProd(
            layer_sizes_branch=[self.Nr**2,100,100,100,64, p], 
            layer_sizes_trunk=[2,100,100,100,64, p],
            activation="relu",
            kernel_initializer="Glorot normal",
        )
        net.to(self.device)  # Ensure the model params live on GPU

        # Compile model
        model = dde.Model(data, net)
        model.compile(
            "adam",
            lr=1e-3,
            loss="mean l2 relative error",
            decay=("inverse time", 1000, 0.5),  # (type, decay_steps, decay_rate)
        )

        print('STARTING MODEL TRAINING')
        # Train model
        losshistory, train_state = model.train(
            iterations=5000,
            display_every=1000,
        )
        
        return model, losshistory, train_state
    
    def c_to_a(self,c):
        '''
        This functions takes in coefficients 'c' and outputs conductivities
        a = exp(sum_i sum_j c_ij 2*sin(i pi x)*sin(j pi y))
        '''
        Ntrunc = c.shape[1]
        y = self.x.clone().detach()
        i = torch.linspace(1,Ntrunc,Ntrunc)
        j = torch.linspace(1,Ntrunc,Ntrunc)
        # [particle q, index i, index j, pos X, pos Y]
        terms = c[:,:,:,None,None]*2*torch.sin(i[None,:,None,None,None]*math.pi*self.x[None,None,None,:,None])*\
                                    torch.sin(j[None,None,:,None,None]*math.pi*y[None,None,None,None,:])
        a = torch.exp(torch.sum(terms,dim=(1,2)))
        
        return a
    
    def c_to_loga(self,c):
        '''
        This functions takes in coefficients 'c' and outputs log-conductivities
        log(a) = sum_i sum_j c_ij 2*sin(i pi x)*sin(j pi y)
        '''
        Ntrunc = c.shape[1]
        y = self.x.clone().detach()
        i = torch.linspace(1,Ntrunc,Ntrunc)
        j = torch.linspace(1,Ntrunc,Ntrunc)
        # [particle q, index i, index j, pos X, pos Y]
        terms = c[:,:,:,None,None]*2*torch.sin(i[None,:,None,None,None]*math.pi*self.x[None,None,None,:,None])*\
                                    torch.sin(j[None,None,:,None,None]*math.pi*y[None,None,None,None,:])
        loga = torch.sum(terms,dim=(1,2))
        
        return loga
    
    def norm(self, a):
        '''
        This function computes the L2 norm of each of the N particles in a distribution. 
        Each particle a_q in a for q=1,...,N is given as a discretized matrix with grid spacing dr
        in both the x and y direction.
        The output is a vector of size N.
        '''
        # Square all elements
        a_squared = a**2  # Shape: (N, Nr_y, Nr_x)

        # Integrate over x (last dimension, columns)
        integral_x = torch.trapezoid(a_squared, dx=self.dr, dim=-1)  # Shape: (N, Nr_y)

        # Integrate over y (now last dimension, rows)
        integral_xy = torch.trapezoid(integral_x, dx=self.dr, dim=-1)  # Shape: (N,)
        
        return torch.sqrt(integral_xy)
    
    def m2(self, a):
        '''
        This function computes the 2nd moment of a distribution made up of N particles. 
        By definition m2(\nu) = integral ||a||^2 d\nu(a)
        Each particle a_q in a for q=1,...,N is given as a discretized matrix with grid spacing dr
        in both the x and y direction.
        '''
        squared_norms = self.norm(a)**2 # Shape: (N,)
        return torch.mean(squared_norms)
    
    def set_parameters(self, Lip, R, mean_test_m2, true_model_zero, c_test, a_test, u_test, a_OOD, u_OOD, a_zero_coef):
        '''
        This function sets important parameters and saves the test distribution.
        '''
        self.Lip = Lip
        self.R = R
        self.mean_test_m2 = mean_test_m2
        self.true_model_zero = true_model_zero
        self.c_test = c_test
        self.a_test = a_test
        self.u_test = u_test
        self.a_OOD = a_OOD
        self.u_OOD = u_OOD
        self.a_zero_coef = a_zero_coef
        
    def get_ot_items(self,a,b,eps=0.1,get_T=True):
        '''
        Computes 2-Wasserstein distance between a and b and evaluates the transport map at a.
        The two distributions a and b are made up of particles.
        These particles are functions given as matrices discretized over the grid.
        '''
        N_a = a.shape[0]
        N_b = b.shape[0]
        # flatten targets (M, d)
        b_flat = b.reshape(N_b, -1) 
        chunk_size = 20
        C = torch.zeros(N_a, N_b, device=self.device)

        for i in range(0, N_a, chunk_size):
            i_end = min(i + chunk_size, N_a)

            # Compute differences for this chunk
            diff = a[i:i_end, None, :, :] - b[None, :, :, :]
            C[i:i_end, :] = self.norm(diff)**2

        train_weights = torch.ones(N_a, device=self.device) / N_a  # uniform weights
        test_weights = torch.ones(N_b, device=self.device) / N_b  # uniform weights

        # Pi = ot.sinkhorn(train_weights, test_weights, C, reg=eps, numItermax=5000)   # Pi shape (N, M)
        Pi = ot.emd(train_weights, test_weights, C)
        W2_sq = torch.sum(Pi * C)
        
        if get_T is False:
            return W2_sq

        # barycentric projection T(x_i) = (1/w_i) (sum_j Pi_ij y_j) / (sum_j Pi_ij)
        row_sums = Pi.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0

        T_flat = (Pi @ b_flat) / row_sums                 # (N_a, d)
        T = T_flat.view_as(a)                             # (N_a, 51, 51)

        return W2_sq, T

    def F(self, c_train, u_train, model):
        '''
        This function computes the objective function evaluated at the coefficients c of the particles 'a'.
        This function takes the coefficients c_train, the true solutions u_train (a->u), and the particles
        a_test which are functions as matrices discretized of the grid.
        '''
        a_train = self.c_to_a(c_train)
        term1 = torch.mean(self.norm(u_train - model.net((a_train.reshape(-1, self.Nr**2),self.trunk)).reshape(-1, self.Nr, self.Nr))**2)

        A = 4*(self.Lip+self.R)**4
        my_pred_zero = model.net((self.a_zero_coef.reshape(-1,self.Nr**2),self.trunk)).reshape(-1,self.Nr,self.Nr)
        my_model_zero = self.norm(my_pred_zero[0])**2
        B = (self.Lip+self.R)**2*16*(self.true_model_zero + my_model_zero)
        D = B + A*self.mean_test_m2

        K = self.a_test.shape[0]
        W2_squared_array = torch.zeros(K, device=self.device)
        for k in range(K):
            W2_squared_array[k] = self.get_ot_items(a_train,self.a_test[k],get_T=False)
        term2 = torch.sqrt(A*self.m2(a_train)+D)*torch.sqrt(torch.mean(W2_squared_array))
        
        return term1 + term2
    
    def computeOODError(self,model):
        K = self.a_OOD.shape[0]
        OOD_errors = torch.zeros(K, device=self.device)
        for k in range(K):
            test_errors = self.norm(self.u_OOD[k] - model.net((self.a_OOD[k].reshape(-1, self.Nr**2),self.trunk)).reshape(-1, self.Nr, self.Nr))**2
            OOD_errors[k] = torch.mean(test_errors)
        mean_OOD_error = torch.mean(OOD_errors)
        return mean_OOD_error
    
    def computeRelativeOODError(self,model):
        K = self.a_OOD.shape[0]
        num_OOD_errors = torch.zeros(K, device=self.device)
        den_OOD_errors = torch.zeros(K, device=self.device)
        for k in range(K):
            num_test_errors = self.norm(self.u_OOD[k] - model.net((self.a_OOD[k].reshape(-1, self.Nr**2),self.trunk)).reshape(-1, self.Nr, self.Nr))**2
            num_OOD_errors[k] = torch.mean(num_test_errors)
            den = self.norm(self.u_OOD[k])**2
        num_mean_OOD_error = torch.mean(num_OOD_errors)
        den_mean = torch.mean(den)
        return num_mean_OOD_error/den_mean
    
    def compute_gradient(self,c_train,a_train,f_train,u_train,model):
        '''
        This function computes the gradient of Df(\nu)(a) evaluated at a_train.
        '''
        NN_a_train = model.net((a_train.reshape(-1, self.Nr**2),self.trunk)).reshape(-1,self.Nr, self.Nr)
        r = u_train - NN_a_train
        w = r.unsqueeze(-1).expand(-1, -1, -1, self.Nt)
        g = self.compute_adjoint(a_train,w)
        adjoint_grad = 2*self.compute_solver_gradient(f_train,g)
        Jr = 2*torch.autograd.grad(NN_a_train, a_train, grad_outputs=r, retain_graph=False)[0]
        term1 = adjoint_grad - Jr

        A = 4*(self.Lip+self.R)**4
        my_pred_zero = model.net((self.a_zero_coef.reshape(-1,self.Nr**2),self.trunk)).reshape(-1,self.Nr,self.Nr)
        my_model_zero = self.norm(my_pred_zero[0])**2
        B = (self.Lip+self.R)**2*16*(self.true_model_zero + my_model_zero)
        D = B + A*self.mean_test_m2

        K = self.a_test.shape[0]
        N = a_train.shape[0]
        W2_squared = torch.zeros(K,device=self.device)
        T = torch.zeros_like(self.a_test)
        for k in range(K):
            W2_squared[k], T[k] = self.get_ot_items(a_train,self.a_test[k])
        mean_W2_squared = torch.mean(W2_squared)
        mean_T = torch.mean(T, dim=0)

        factor = torch.sqrt(mean_W2_squared/(A*self.m2(a_train)+D))
        term2 = A*a_train*factor

        term3 = (a_train-mean_T)/factor
        
        grad_a = term1+term2+term3
        # Chain Rule to get gradient wrt c
        grad_c = torch.autograd.grad(a_train, c_train, grad_outputs=grad_a, retain_graph=False)[0]

        return grad_c        