import torch
from util.utilities_module import vec_to_cholesky, process_cholesky, solve_with_cholesky, sqrtmh
from kernels import gaussian_kernel
from quadrature import rpcholesky


class Model(object):
    """
    Kernel ridge regression model and adjoint state vector solver for bilevel
        constrained optimization over RKHS.
    data_test is a list of tuples of "test" pairs (V_j, f_true(V_j)).
    """
    def __init__(self,
                 data_train,    # train model
                 X_val,         # adjoint solve and MC sum
                 data_test,     # adjoint RHS
                 kernel=gaussian_kernel, 
                 eps=1e-10,
                 device=None
                 ):
        super().__init__()
        
        self.data_test = data_test
        self.kernel = kernel
        self.eps = eps
        self.device = device
    
        self.X_train, self.Y_train = data_train
        self.X_val = X_val
        self.N_val = self.X_val.shape[0]
        self.num_test_tuples = len(self.data_test)
        
        self.coeff = None
        self.adjoint_values = None
        
        self.train()
        
    def _make_reg_kernel_matrix(self, X):
        N = X.shape[0]
        X = X.to(self.device)
        X = self.kernel(X, X)
        X += N*self.eps*torch.eye(N, device=self.device)
        return X
        
    def train(self):
        KXX_chol = self._make_reg_kernel_matrix(self.X_train)
        KXX_chol = torch.linalg.cholesky(KXX_chol)
        self.coeff = solve_with_cholesky(KXX_chol, self.Y_train.to(self.device))
        
    def form_RHS(self):
        """
        Form RHS for adjoint solve.
        """
        X = self.X_val.to(self.device)
        RHS = torch.zeros(self.N_val, device=self.device)
        for data_tuple in self.data_test:
            V, Y = data_tuple
            V = V.to(self.device)
            Y = Y.to(self.device)
            Y -= self.forward(V)
            V = self.kernel(X, V)
            V = torch.einsum('ij,j->i', V, Y) 
            RHS += V / Y.shape[0]
            
        return self.N_val * RHS / self.num_test_tuples
    
    def solve_adjoint_values(self):
        rhs = self.form_RHS()
        dev = rhs.device
        KXX_chol = self._make_reg_kernel_matrix(self.X_val)
        KXX_chol = torch.linalg.cholesky(KXX_chol)
        self.adjoint_values = solve_with_cholesky(KXX_chol.to(dev), rhs)
            
    def get_adjoint_values(self):
        """
        Return adjoint state values at the training data points
        """
        return self.adjoint_values

    def forward(self, x):
        """
        x: (nbatch, d) tensor, where d is the input dimension.
        Returns (nbatch,) tensor of scalar predictions.
        """
        dev = x.device
        x = self.kernel(x, self.X_train.to(dev))
        return torch.einsum('ij,j->i', x, self.coeff.to(dev))
        
    def __call__(self, x):
        return self.forward(x)


class DesignDistribution(object):
    """
    Gaussian distribution with fixed mean and variable covariance matrix.
    """
    def __init__(self,
                 mean,
                 device=None
                 ):
        super().__init__()
        
        self.device = device
        self.mean = mean.to(device)
        
        # Initialize
        self.d = self.mean.shape[0]
        self.theta = process_cholesky(torch.eye(self.d, device=self.device), self.d)
        
        # TODO: debug initialization (away from identity Chol)
        # Z = torch.randn(self.d, self.d, device=self.device)
        # self.theta = process_cholesky(torch.tril(Z.T@Z), self.d)

        self.distribution = None
        self.set_params(self.theta)

    def set_params(self, theta, FLAG_PROCESS=True):
        if FLAG_PROCESS:
            theta = vec_to_cholesky(theta, self.d)
            theta = process_cholesky(theta, self.d)
        self.theta = theta
        chol = vec_to_cholesky(self.theta, self.d) 
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, 
                                                                        scale_tril=chol)
    def neg_log_prob(self, x, theta):
        """
        x: (nbatch, d) tensor
        Returns (nbatch,) tensor
        """
        # Misfit
        z, chol = self.solve_misfit(x, theta, return_chol=True)
        
        # Logdet
        z += torch.sum(torch.log(torch.diagonal(chol)))
        
        return z
    
    def solve_misfit(self, x, theta, return_chol=False):
        """
        x: (nbatch, d) tensor
        Returns (nbatch,) tensor
        """
        x = x - self.mean
        chol = vec_to_cholesky(theta, self.d)
        z = solve_with_cholesky(chol, x.permute(1,0)).permute(1,0)
        z = torch.sum(x*z, dim=-1) * 0.5
        
        if return_chol:
            z = (z, chol)
        
        return z
    
    def parametric_score(self, x, theta):
        """
        x: (nbatch, d) tensor
        Returns (nbatch, dim_theta) tensor (full Jacobian) representing grad(log(density))
        """
        # Misfit Jacobian
        x = torch.func.jacfwd(self.solve_misfit, argnums=1)(x, theta)
        
        # Logdet gradient
        theta.requires_grad_(True)
        grad_ld = vec_to_cholesky(theta, self.d)
        grad_ld = torch.sum(torch.log(torch.diagonal(grad_ld)))
        grad_ld = torch.autograd.grad(grad_ld, theta)[0]
        
        return -(x + grad_ld)
    
    def sample(self, n=1):
        return self.distribution.sample((n,)).to(self.device)
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)
    
    
class DesignDistribution_WithMean(object):
    """
    Gaussian distribution with variable mean and covariance matrix.
    """
    def __init__(self,
                 d,
                 mean_init=None,
                 device=None
                 ):
        super().__init__()
        
        self.d = d
        self.mean_init = mean_init
        self.device = device
        
        # Initialize
        self.theta = process_cholesky(torch.eye(self.d, device=self.device), self.d) # Id cov initialization
        if self.mean_init is None:
            self.mean_init = torch.zeros(self.d)
        self.theta = self.from_pair(self.mean_init.to(self.device), self.theta) # zero mean initialization

        self.distribution = None
        self.set_params(self.theta)
        
    def from_pair(self, mean, vchol):
        return torch.cat((mean, vchol))
    
    def to_pair(self, theta):
        """
        First self.d entries of theta is mean; remaining is cholesky vector
        """
        return theta[:self.d], theta[self.d:]

    def set_params(self, theta, FLAG_PROCESS=True):
        mean, vchol = self.to_pair(theta)
        if FLAG_PROCESS:
            vchol = vec_to_cholesky(vchol, self.d)
            vchol = process_cholesky(vchol, self.d)
        self.theta = self.from_pair(mean, vchol)
        chol = vec_to_cholesky(vchol, self.d) 
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, 
                                                                        scale_tril=chol)
    def neg_log_prob(self, x, theta):
        """
        x: (nbatch, d) tensor
        Returns (nbatch,) tensor
        """        
        # Misfit
        z, chol = self.solve_misfit(x, *self.to_pair(theta), return_chol=True)
        
        # Logdet
        z += torch.sum(torch.log(torch.diagonal(chol)))
        
        return z

    def _solve_shift(self, x, mean, vchol):
        x = x - mean
        chol = vec_to_cholesky(vchol, self.d)
        z = solve_with_cholesky(chol, x.permute(1,0)).permute(1,0)
        return (x, chol, z)
    
    def solve_misfit(self, x, mean, vchol, return_chol=False):
        """
        x: (nbatch, d) tensor
        Returns (nbatch,) tensor
        """
        x, chol, z = self._solve_shift(x, mean, vchol)
        z = torch.sum(x*z, dim=-1) * 0.5
        
        if return_chol:
            z = (z, chol)
        
        return z
    
    def parametric_score(self, x, theta):
        """
        x: (nbatch, d) tensor
        Returns (nbatch, dim_theta) tensor (full Jacobian) representing grad(log(density))
        """
        mean, vchol = self.to_pair(theta)

        # Gradient with respect to mean
        _, _, grad_m = self._solve_shift(x, mean, vchol)
        
        # Misfit Jacobian with respect to vchol
        x = torch.func.jacfwd(self.solve_misfit, argnums=2)(x, mean, vchol)
        
        # Logdet gradient
        vchol.requires_grad_(True)
        grad_ld = vec_to_cholesky(vchol, self.d)
        grad_ld = torch.sum(torch.log(torch.diagonal(grad_ld)))
        grad_ld = torch.autograd.grad(grad_ld, vchol)[0]
        
        return torch.cat((grad_m, -x - grad_ld), dim=-1)
    
    def sample(self, n=1):
        return self.distribution.sample((n,)).to(self.device)
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)


class UniformDistribution(object):
    """
    Uniform distribution on [low, high]^d
    """
    def __init__(self,
                 d,
                 low=0,
                 high=1,
                 device=None
                 ):
        super().__init__()
        
        self.d = d
        self.low = low
        self.high = high
        self.device = device

        self.distribution = torch.distributions.uniform.Uniform(self.low, self.high)
    
    def sample(self, n=1):
        return self.distribution.sample((n, self.d)).to(self.device)
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)
    
class RPCholeskyDistribution(object):
    """
    RPCholesky
    """
    def __init__(self,
                 kernel,
                 proposal,
                 device=None
                 ):
        super().__init__()

        self.kernel = kernel
        self.proposal = proposal
        self.device = device
    
    def sample(self, n=1):
        return rpcholesky(self.proposal, n, self.kernel)
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)

class CallableDistribution(object):
    """
    Turns a torch.distribution into a callable class object
    """
    def __init__(self,
                 distribution,
                 device=None
                 ):
        super().__init__()
        
        self.distribution = distribution
        self.device = device
    
    def sample(self, n=1):
        return self.distribution.sample((n,)).to(self.device)
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)


class FiniteMixture(object):
    """
    Finite mixture model
        distribution_list: (list), list of torch.distributions
    """
    def __init__(self,
                 distribution_list,
                 weights=None,
                 device=None
                 ):
        super().__init__()
        
        self.distribution_list = distribution_list
        self.weights = weights
        self.device = device
        
        self.num_distributions = len(self.distribution_list)
        if self.weights is None:
            self.weights = torch.ones(self.num_distributions) / self.num_distributions
            
        assert len(self.distribution_list) == len(self.weights)
    
    def sample(self, n=1):
        indices = torch.multinomial(self.weights, n, replacement=True)
        out = []
        for i in indices:
            out.append(self.distribution_list[i].sample().to(self.device))
            
        out = torch.stack(out)
        
        return out
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)


class GaussianBarycenter(object):
    """
    Wasserstein-2 barycenter (with equal weights) of a finite collection of Gaussian distributions

        means: (N, d) tensor
        covariances: (N, d, d) tensor
        C0: (d, d) tensor, initial guess for barycenter covariance
    """
    def __init__(self,
                 means,
                 covariances,
                 C0=None,
                 device=None
                 ):
        super().__init__()
        
        self.means = means
        self.covariances = covariances
        self.C0 = C0
        self.device = device
        
        # Barycenter mean (mean of means)
        self.bmean = torch.mean(means, dim=0)

        # Initialize
        self.distribution = None
        self.bcov = None
        self.d = self.bmean.shape[-1]
        self.N = self.covariances.shape[0]
        if self.C0 is None:
            self.C0 = torch.eye(self.d).to(self.device)     # identity matrix
        
        self.fixed_point_solver()
        self.set_params(self.bmean, self.bcov)

    def set_params(self, mean, cov):
        mean = mean.to(self.device)
        cov = cov.to(self.device)
        self.distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, 
                                                                        covariance_matrix=cov)
    
    def _rel_error_fro(self, A, Atrue):
        return torch.linalg.matrix_norm(A-Atrue) / torch.linalg.matrix_norm(Atrue)
    
    def fixed_point_solver(self, max_iter=50, rtol=1e-6, VERBOSE=False):
        C_old = self.C0.to(self.device)
        err = 1.0
        my_iter = 0
        while my_iter <= max_iter and err > rtol:
            C = 0.0 * C_old
            root = sqrtmh(C_old)
            for i in range(self.N):
                temp = self.covariances[i,...].to(self.device) @ root
                temp = root @ temp
                temp = sqrtmh(temp)
                C += temp
                
            C /= self.N
            
            # Update
            my_iter += 1
            err = self._rel_error_fro(C, C_old)
            C_old = C
            
        # Return approximate barycenter covariance
        if VERBOSE:
            print(f'Final iteration [{my_iter}/{max_iter}], Rel Error: {err}')
        self.bcov = C_old
        
    def sample(self, n=1):
        return self.distribution.sample((n,)).to(self.device)
        
    def __call__(self, *args,  **kwargs):
        return self.sample(*args, **kwargs)


def meta_loss(model, data_test, device=None, TAKE_ROOT=True, USE_RELATIVE=True):
    if USE_RELATIVE:
        loss = 0
        loss_gt = 0
        for data_tuple in data_test:
            V, Y = data_tuple
            V = V.to(device)
            Y = Y.to(device)
            loss_gt += torch.mean(Y**2)
            Y -= model(V)
            loss += torch.mean(Y**2)
            
        loss /= loss_gt
    else:
        loss = 0
        for data_tuple in data_test:
            V, Y = data_tuple
            V = V.to(device)
            Y = Y.to(device)
            Y -= model(V)
            loss += torch.mean(Y**2)

        loss = loss / len(data_test)
    
    if TAKE_ROOT:
        return torch.sqrt(loss)
    else:
        return loss


def model_update(N, design, truth, eps, data_test, kernel, device, return_data=False, N_train_percent=None, split_data=True):   
    # Sample training and validation data from current design
    X_val = design(N)
    Y_val = truth(X_val)
    
    # Split dataset to avoid overfitting which slows down gradient descent
    if split_data:
        if N_train_percent is None:
            N_train_percent = 66.66
        
        N_train = int(N_train_percent / 100 * N)

        if N <= N_train:
            raise ValueError("N must be greater than N_train")
            
        data_train = (X_val[:N_train, ...], Y_val[:N_train, ...])
        X_val = X_val[N_train:, ...]
        Y_val = Y_val[N_train:, ...]
    else:
        data_train = (X_val, Y_val)
    
    # Train model and solve adjoint for current design
    model = Model(data_train, X_val, data_test, kernel=kernel, eps=eps, device=device)
    model.solve_adjoint_values()
    
    if return_data:
        model = (model, X_val, Y_val)
    
    return model