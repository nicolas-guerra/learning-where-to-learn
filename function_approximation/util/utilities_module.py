import torch
import math
import numpy as np
import warnings


class CosineAnnealingLR(object):
    """
    Calculates the learning rate based on cosine annealing.

    Args:
        current_step: The current training step (or epoch).
        T_max: The maximum number of steps (or epochs).
        initial_lr: The initial learning rate.
        eta_min: The minimum learning rate.
    """
    def __init__(self,
                 T_max,
                 initial_lr,
                 eta_min=0.0
                 ):
        super().__init__()
        
        self.T_max = T_max
        self.initial_lr = initial_lr
        self.eta_min = eta_min
    
    def get_lr(self, current_step):
        lr = self.eta_min + (self.initial_lr - self.eta_min) * 0.5 * \
            (1. + math.cos(math.pi * current_step / self.T_max))
        return lr
    
    def __call__(self, *args,  **kwargs):
        return self.get_lr(*args, **kwargs)

def to_torch(x, to_float=True):
    """
    Send input numpy array to single precision torch tensor if to_float=True, else double.
    """
    if to_float:
        if np.iscomplexobj(x):
            x = x.astype(np.complex64)
        else:
            x = x.astype(np.float32)
    return torch.from_numpy(x)

def get_emp_mean(X):
    """
    X: dataset of size (N, d)
    """
    return torch.mean(X, dim=0)

def get_emp_cov(X):
    """
    X: dataset of size (N, d)
    """
    m = get_emp_mean(X)
    X = X - m
    return X.T@X / X.shape[0]

def sqrtmh(A):
    """
    Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices
    https://github.com/pytorch/pytorch/issues/25481#issuecomment-1032789228

    """
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return (Q * L.sqrt().unsqueeze(-2)) @ Q.mH

def solve_with_cholesky(L_chol, b):
    """
    L_chol is square lower triangular matrix, (..., d, d)
    b is right hand side, (..., d, k) where k geq 1
    """
    if b.ndim < 2:
        b = b[..., None]
    x = torch.linalg.solve_triangular(L_chol, b, upper=False)
    x = torch.linalg.solve_triangular(L_chol.T.conj(), x, upper=True)
    return x.squeeze()

def vec_to_cholesky(vec, d):
    """
    Map from reduced representation of cholesky factor (scalar, diag, or lower entries) to full matrix:
        Scalar to Scalar times d by d identity
        d-dim vector to diagonal matrix
        N=d(d+1)/2 entries of LOWER cholesky factor to d by d lower cholesky factor matrix itself
    """
    device = vec.device
    vec = torch.squeeze(vec)
    if vec.ndim == 1:
        if vec.shape[-1] == d:
            chol = torch.diag(vec)
        elif 2*vec.shape[-1] == d * (d + 1):
            chol = torch.zeros((d,d), device=device)
            chol[*torch.tril_indices(d, d, device=device)] = vec
        else:
            raise ValueError('cholesky vector must have d or d(d+1)/2 entries!')
    elif vec.ndim == 0: # scalar times identity
        chol = vec*torch.eye(d, device=device)
    elif vec.ndim >= 2:
        raise ValueError('vec must have at most one dimension.')

    return chol

def cholesky_to_vec(chol):
    """
    Map full lower triangular matrix to vector representation of lower Cholesky factor
    """
    return chol[*torch.tril_indices(*chol.shape[-2:], device=chol.device)]

def torch_fill_diagonal(dest_matrix, source_vector):
    """
    https://stackoverflow.com/questions/49429147/replace-diagonal-elements-with-vector-in-pytorch
    """
    dest_matrix[range(min(dest_matrix.size())), range(min(dest_matrix.size()))] = source_vector
    
def process_cholesky(chol, N, FLAG_constraints=True, check_valid='warn', FLAG_POS=True, eps=1e-7):
    """
    Process given cholesky factor into compressed vector representation
    Args
    ----
        chol : 0-D or 1-D or 2-D array_like, for cholesky factor L
            Lower cholesky factor of the output covariance matrix LL*. It must be symmetric and
            positive-semidefinite for proper sampling. It must be strictly positive definite for
            invertibility.
            - 0-D or float means float times the identity is the Cholesky factor
            - 1-D means diag(chol) or N*(N+1)//2 entries
            - 2-D means generic lower triangular cholesky factor with positive
                diagonal entries
        N : (int): dimension of output space
        check_valid : { 'warn', 'raise', 'ignore' }, optional
            Behavior when the covariance matrix is not positive semidefinite.
    """
    device = chol.device
    Nchol = N*(N + 1)//2 # Nchol = 1 + 2 + ... + N
    chol = torch.squeeze(chol)
    CTYPE = None
    if chol.ndim > 2:
        raise ValueError('chol must have at most two dimensions. Scalar is also okay.')
    elif chol.ndim == 2: # check if lower triangular
        if (chol.shape[0] != chol.shape[1]) or not torch.allclose(chol, torch.tril(chol)):
            raise ValueError('2d input must be square lower triangular')
    elif chol.ndim == 1:
        if chol.shape[-1] == N:
            CTYPE = 1
            chol = torch.diag(chol)
        elif chol.shape[-1] == Nchol:
            chol = vec_to_cholesky(chol, d=N)
        else:
            raise ValueError('cholesky vector must have N or N(N+1)/2 entries!')
    elif chol.ndim == 0: # scalar times identity
        CTYPE = 0
        chol = chol*torch.eye(N, device=device)
    else:
        raise ValueError('chol is an invalid data type.')
    
    if FLAG_constraints:
        # Enforce PSD
        torch_fill_diagonal(chol, torch.maximum(torch.tensor(0, device=device), torch.diag(chol)))

        # Enforce strictly positive by replacing zero diagonal entries with eps
        # https://stackoverflow.com/a/66760667
        if FLAG_POS:
            mask = (chol.diagonal() == 0)
            chol += torch.diag(eps*mask)
    else:
        if check_valid != 'ignore':
            if check_valid != 'warn' and check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")
            psd = (torch.sum(torch.diag(chol) < 0) == 0)
            if not psd:
                if check_valid == 'warn':
                    warnings.warn("cholesky diagonal is not >=0:\
                                  covariance is not positive-semidefinite.",
                        RuntimeWarning)
                else:
                    raise ValueError(
                        "cholesky diagonal is not >=0: covariance is not positive-definite.")
    
    # Return compression
    if CTYPE == 0:
        chol = chol[0,0] # scalar
    elif CTYPE == 1:
        chol = torch.diagonal(chol) # diagonal
    else:
        chol = cholesky_to_vec(chol)
    return chol

