import torch
from util.utilities_module import vec_to_cholesky, process_cholesky, solve_with_cholesky
from kernels import gaussian_kernel


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


class Model(object):
    """
    Kernel ridge regression model and adjoint state vector solver for bilevel
        constrained optimization over RKHS.
    data_test is a list of tuples of "test" pairs (V_j, f_true(V_j)).
    """
    def __init__(self,
                 data,
                 data_test,
                 kernel=gaussian_kernel, 
                 eps=1e-10,
                 device=None
                 ):
        super().__init__()
        
        self.data = data
        self.data_test = data_test
        self.kernel = kernel
        self.eps = eps
        self.device = device
    
        self.X, self.Y = self.data
        self.N = self.X.shape[0]
        self.num_test_tuples = len(self.data_test)
        self.KXX_chol = None
        self.KXX = None
        self.coeff = None
        self.adjoint_values = None
        self.model_values = None
        
        self.train()
        self.solve_adjoint_values()
        self.solve_model_values()
        
    def train(self):
        X = self.X.to(self.device)
        Y = self.Y.to(self.device)
        X = self.kernel(X, X)
        self.KXX = X.clone().detach().cpu()
        X += self.N*self.eps*torch.eye(self.N, device=self.device)
        self.KXX_chol = torch.linalg.cholesky(X)
        self.coeff = solve_with_cholesky(self.KXX_chol, Y)
        
    def form_RHS(self):
        """
        Form RHS for adjoint solve.
        """
        X = self.X.to(self.device)
        RHS = torch.zeros(self.N, device=self.device)
        for data_tuple in self.data_test:
            V, Y = data_tuple
            V = V.to(self.device)
            Y = Y.to(self.device)
            Y -= self.forward(V)
            V = self.kernel(X, V)
            V = torch.einsum('ij,j->i', V, Y) 
            RHS += V / Y.shape[0]
            
        return self.N * RHS / self.num_test_tuples
    
    def solve_adjoint_values(self):
        rhs = self.form_RHS()
        dev = rhs.device
        self.adjoint_values = solve_with_cholesky(self.KXX_chol.to(dev), rhs)
            
    def get_adjoint_values(self):
        """
        Return adjoint state values at the training data points
        """
        return self.adjoint_values
    
    def solve_model_values(self):
        dev = self.coeff.device
        self.model_values = torch.einsum('ij,j->i', self.KXX.to(dev), self.coeff)
        
    def get_model_values(self):
        """
        Return model values at the training data points
        """
        return self.model_values

    def forward(self, x):
        """
        x: (nbatch, d) tensor, where d is the input dimension.
        Returns (nbatch,) tensor of scalar predictions.
        """
        dev = x.device
        x = self.kernel(x, self.X.to(dev))
        return torch.einsum('ij,j->i', x, self.coeff.to(dev))
        
    def __call__(self, x):
        return self.forward(x)


class Adjoint(object):
    """
    Adjoint state update as a full function. data_test is a list of tuples of pairs (V_j, f_true(V_j)-f_model(V_j)).
    """
    def __init__(self,
                 X_train,
                 data_test,
                 kernel=gaussian_kernel, 
                 eps=1e-12,
                 device=None
                 ):
        super().__init__()
        
        self.X_train = X_train
        self.data_test = data_test
        self.kernel = kernel
        self.eps = eps
        self.device = device
    
        self.N = self.X_train.shape[0]
        self.num_test_tuples = len(self.data_test)
        self.coeff = None
        
        self.train()
        
    def form_RHS(self):
        X = self.X_train.to(self.device)
        RHS = torch.zeros(self.N, device=self.device)
        for data_tuple in self.data_test:
            V, Y = data_tuple
            V = V.to(self.device)
            Y = Y.to(self.device)
            V = self.kernel(X, V)
            V = torch.einsum('ij,j->i', V, Y) 
            RHS += V / Y.shape[0]
            
        return RHS / self.num_test_tuples
            
    def train(self):
        X = self.X_train.to(self.device)
        KXX = self.kernel(X, X)
        KXX = torch.mm(KXX, KXX)/self.N
        KXX += self.eps*torch.eye(self.N, device=self.device)
        self.coeff = torch.linalg.solve(KXX, self.form_RHS())

    def forward(self, x):
        """
        x: (nbatch, d) tensor, where d is the input dimension
        """
        dev = x.device
        x = self.kernel(x, self.X_train.to(dev))
        return torch.einsum('ij,j->i', x, self.coeff.to(dev))
        
    def __call__(self, x):
        return self.forward(x)


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
    
    
def model_update(N, design, truth, eps, data_test, kernel, device, return_data=False):
    # Sample training data from current design
    X = design(N)
    Y = truth(X)
    data = (X, Y)

    # Build model and solve adjoint for current design
    model = Model(data, data_test, kernel=kernel, eps=eps, device=device)
    
    if return_data:
        model = (model, X, Y)
    
    return model
