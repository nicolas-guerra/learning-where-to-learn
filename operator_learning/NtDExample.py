import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags, kron, eye
import matplotlib.pyplot as plt
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
from deepxde.nn import DeepONet
from deepxde.nn.pytorch.deeponet import DeepONetCartesianProd
from scipy.optimize import differential_evolution
import pandas as pd
from scipy.fftpack import idst
import pickle

# True Operator for DtN
def getAx(a):
    # a is (M, M)
    # horizontal harmonic mean on each interior edge
    Ah = a[:, :-1] + a[:, 1:]
    # avoid dividing by 0: if sum ==0, set harmonic mean to 0
    hm = np.where(Ah == 0,
                  0.0,
                  2 * a[:, :-1] * a[:, 1:] / Ah)
    # flatten row‐major, length = M*(M-1)
    return diags(hm.ravel())

def getAy(a):
    # vertical edges
    Av = a[:-1, :] + a[1:, :]
    hm = np.where(Av == 0,
                  0.0,
                  2 * a[:-1, :] * a[1:, :] / Av)
    return diags(hm.ravel())

def neumann_flux(u, a):
    """
    Compute the Neumann flux q = a * du/dn on the boundary of an M×M grid,
    including corners (using the average flux in vertical and horizontal dir.), and return a 1D array
    of length 4*(M-1) going clockwise starting at the top‐left corner.
    """
    M = u.shape[0]
    assert u.shape == (M, M) and a.shape == (M, M)
    h = 1.0 / (M - 1)
    flux_vals = []

    # Top‐left corner (i=0,j=0)
    dudn_vert = (u[0, 0] - u[1, 0]) / h      # normal points upward
    dudn_horz = (u[0, 0] - u[0, 1]) / h      # normal points leftward
    flux_vals.append(a[0, 0] * 0.5 * (dudn_vert + dudn_horz))

    # Top edge interior (i=0, j=1..M-2)
    for j in range(1, M - 1):
        dudn = (u[0, j] - u[1, j]) / h
        flux_vals.append(a[0, j] * dudn)

    # Top‐right corner (i=0,j=M-1)
    dudn_vert = (u[0, M - 1] - u[1, M - 1]) / h
    dudn_horz = (u[0, M - 1] - u[0, M - 2]) / h
    flux_vals.append(a[0, M - 1] * 0.5 * (dudn_vert + dudn_horz))

    # Right edge interior (j=M-1, i=1..M-2)
    for i in range(1, M - 1):
        dudn = (u[i, M - 1] - u[i, M - 2]) / h
        flux_vals.append(a[i, M - 1] * dudn)

    # Bottom‐right corner (i=M-1,j=M-1)
    dudn_vert = (u[M - 1, M - 1] - u[M - 2, M - 1]) / h  # normal points downward
    dudn_horz = (u[M - 1, M - 1] - u[M - 1, M - 2]) / h  # normal points rightward
    flux_vals.append(a[M - 1, M - 1] * 0.5 * (dudn_vert + dudn_horz))

    # Bottom edge interior (i=M-1, j=M-2..1)
    for j in range(M - 2, 0, -1):
        dudn = (u[M - 1, j] - u[M - 2, j]) / h
        flux_vals.append(a[M - 1, j] * dudn)

    # Bottom‐left corner (i=M-1,j=0)
    dudn_vert = (u[M - 1, 0] - u[M - 2, 0]) / h
    dudn_horz = (u[M - 1, 0] - u[M - 1, 1]) / h
    flux_vals.append(a[M - 1, 0] * 0.5 * (dudn_vert + dudn_horz))

    # Left edge interior (j=0, i=M-2..1)
    for i in range(M - 2, 0, -1):
        dudn = (u[i, 0] - u[i, 1]) / h
        flux_vals.append(a[i, 0] * dudn)

    return np.array(flux_vals)

def DtN(dbc,a,f=None):
    """
    Solve -div(a grad u) = f on [0,1]^2 with Dirichlet boundary condition
    a: (N+2, N+2) array, permeability field
    f: (N+2, N+2) array, source term (default = 1), Be sure the boundary of f is 0!
    Let M=N+2. This function takes N+2xN+2 matrix 'a' which describes the permeability of an MxM grid. 
    This function then outputs the neumann boundary condition from the solution.
    """
    
    M = a.shape[0]
    N = M-2
    h = 1/(M-1)

    # Flatten f
    if f is None:
        f = np.zeros((M,M))
    # Put dbc around f
    f[0,0:M] = dbc[:M]
    f[1:M,M-1] = dbc[M:2*M - 1]
    f[M-1,M-2::-1] = dbc[2*M-1:3*M-2]
    f[M-2:0:-1, 0] = dbc[3*M-2:]
    f_flat = f.reshape(M**2)
    
    # Flux matrices
    Ax = getAx(a)
    Ay = getAy(a)

    # Derivative operators
    e = np.ones(M)
    D = diags([-e, e], [0,1], shape=(M-1, M)) / h
    I = eye(M)
    Dx = kron(I,D)
    Dy = kron(D,I)

    # Construct full operator
    L = Dx.T @ Ax @ Dx + Dy.T @ Ay @ Dy
    # Pin boundary coefficients to 1 and the rest to 0 in that row to satisfy DBC
    L = L.tolil()
    for i in range(M):
        for j in range(M):
            if i==0 or i==M-1 or j==0 or j==M-1:
                idx = i*M + j
                L[idx, :]   = 0
                L[idx, idx] = 1
    L = L.tocsr()

    # Solve Linear system
    u_flat = spsolve(L,f_flat)
    u_full = u_flat.reshape((M, M)) 
    nbc = neumann_flux(u_full, a)
    nbc -= np.mean(nbc) # Enforce NBC integrates to 0
    return nbc, L

def DtNwithL(dbc,L):
    """
    Solve -div(a grad u) = f on [0,1]^2 with Dirichlet boundary condition
    a: (N+2, N+2) array, permeability field
    f: (N+2, N+2) array, source term (default = 1), Be sure the boundary of f is 0!
    Let M=N+2. This function takes N+2xN+2 matrix 'a' which describes the permeability of an MxM grid. 
    This function then outputs the neumann boundary condition from the solution.
    """
    
    M = a.shape[0]
    N = M-2
    h = 1/(M-1)

    # Flatten f
    f = np.zeros((M,M))
    # Put dbc around f
    f[0,0:M] = dbc[:M]
    f[1:M,M-1] = dbc[M:2*M - 1]
    f[M-1,M-2::-1] = dbc[2*M-1:3*M-2]
    f[M-2:0:-1, 0] = dbc[3*M-2:]
    f_flat = f.reshape(M**2)
    
    # Solve Linear system
    u_flat = spsolve(L,f_flat)
    u_full = u_flat.reshape((M, M)) 
    nbc = neumann_flux(u_full, a)
    nbc -= np.mean(nbc) # Enforce NBC integrates to 0
    return nbc

def geta_samples_dst(M, tau, sigma=None, alpha=2, N_samples=1, z=None):
    """
    Returns N_samples draws of a log‐Gaussian field on an M×M grid including boundaries.
    Uses a sine‐basis (zero at the boundaries), then exponentiates.

    Args:
      M         : total grid size in each dim (including boundary points)
      tau       : length‐scale parameter
      sigma     : marginal std (if None, defaults to tau**(alpha-1))
      alpha     : spectral exponent
      N_samples : number of independent samples to draw
      z         : optional pre‐drawn standard normals of shape (N_samples, M-2, M-2)

    Returns:
      a : ndarray of shape (N_samples, M, M), where
          - interior points follow ∑₍i,j₎ cᵢⱼ sin(iπx)sin(jπy)
          - boundary is exp(0)=1
    """
    # 1) number of interior points
    N = M - 2
    if N <= 0:
        raise ValueError("M must be ≥ 2 so that M-2 ≥ 1 interior points.")
    
    # 2) default sigma
    if sigma is None:
        sigma = tau**(alpha - 1)
    
    # 3) build spectral scaling λ_{i,j} on interior modes
    i = np.arange(1, N+1)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    lam = sigma / (((ii*np.pi)**2 + (jj*np.pi)**2 + tau**2)**(alpha/2))
    
    # 4) draw (or use) standard normals
    if z is None:
        z = np.random.randn(N_samples, N, N)
    coeffs = z * lam[None, :, :]  # shape (N_samples, N, N)
    
    # 5) invert via 2-D DST on interior (axes 1 & 2)
    #    SciPy’s dst(type=2, norm='ortho') is self‐inverse
    u_int = idst(coeffs, axis=1, type=2, norm='ortho')
    u_int = idst(u_int,   axis=2, type=2, norm='ortho')  # shape (N_samples, N, N)
    
    # 6) pad with zeros on the boundary so u=0 at edges
    u_full = np.zeros((N_samples, M, M), dtype=u_int.dtype)
    u_full[:, 1:-1, 1:-1] = u_int
    
    # 7) exponentiate: boundary becomes exp(0)=1
    return np.exp(u_full)

def getDBCsamples(num_pts, tau, sigma=None, alpha=2, N_samples=1, z=None):
    # default sigma
    if sigma is None:
        sigma = tau**(alpha - 1/2)

    # Get eigenvalues
    i = np.arange(1, num_pts+1)
    lam = sigma / (((i * np.pi)**2 + tau**2)**(alpha/2))  # shape (N,)

    # Draw or use provided standard normals
    if z is None:
        z = np.random.randn(N_samples, num_pts)
    coeffs = z * lam[None, :]  # shape (N_samples, N)

    # Inverse DST (type-II, orthonormal) to reconstruct DBC
    #    SciPy idst(type=2, norm='ortho') is the inverse transform
    dbc = idst(coeffs, axis=1, type=2, norm='ortho')  # (N_samples, N)
    return dbc

def getNBCsamples(num_pts, tau, sigma=None, alpha=2, N_samples=1, z=None):
    # default sigma
    if sigma is None:
        sigma = tau**(alpha - 1/2)

    # Get eigenvalues
    i = np.arange(1, num_pts+1)
    lam = sigma / (((i * np.pi)**2 + tau**2)**(alpha/2))  # shape (N,)

    # Draw or use provided standard normals
    if z is None:
        z = np.random.randn(N_samples, num_pts)
    coeffs = z * lam[None, :]  # shape (N_samples, N)

    # Inverse DST (type-II, orthonormal) to reconstruct DBC
    #    SciPy idst(type=2, norm='ortho') is the inverse transform
    nbc = idst(coeffs, axis=1, type=2, norm='ortho')  # (N_samples, N)
    nbc -= np.mean(nbc,axis=1)[:,None] # Enforcing NBC integrates to 0 
    return nbc

def getNBCsamples_MeanShift(num_pts, omega, tau=3, sigma=None, alpha=2, N_samples=1, z=None):
    # default sigma
    if sigma is None:
        sigma = tau**(alpha - 1/2)

    # Get sqrt eigenvalues
    i = np.arange(1, num_pts+1)
    lam = sigma / (((i * np.pi)**2 + tau**2)**(alpha/2))  # shape (N,)

    # Draw or use provided standard normals
    if z is None:
        z = np.random.randn(N_samples, num_pts)
    coeffs = z * lam[None, :]  # shape (N_samples, N)
    
    # Mean Shift
    x = np.linspace(0,1,num_pts)
    mean_shift = np.sin(omega*(x-0.5))
    
    # Inverse DST (type-II, orthonormal) to reconstruct DBC
    #    SciPy idst(type=2, norm='ortho') is the inverse transform
    nbc = 10*mean_shift + idst(coeffs, axis=1, type=2, norm='ortho')  # (N_samples, N)
    nbc -= np.mean(nbc,axis=1)[:,None] # Enforcing NBC integrates to 0 
    return nbc

def build_D(M, a):
    """
    Build the DtN matrix D of shape (4*M-4, 4*M-4), so that
      nbc = D @ dbc
    
    M : int, grid size (including boundary)
    a : (M,M) conductivity array
    """
    print('Constructing NtD Operator')
    n_rows = 4*M - 4       # number of Neumann outputs
    n_cols = 4*M - 4       # length of full DBC
    D = np.zeros((n_rows, n_cols), dtype=float)

    # basis e_j in the full-DBC space
    for j in range(n_cols):
        e = np.zeros(n_cols, dtype=float)
        e[j] = 1.0
        # DtN knows how to take a full-length dbc and return n_rows fluxes
        if j == 0:
            D[:, j], L = DtN(e, a, f=None)
        else:
            D[:, j] = DtNwithL(e, L)
        if j % 10 == 0:
            print(f"Constructed column {j+1}/{n_cols}")
    return D

# Fix conductivity and construct D
M = 128
tau_a = 3 # This tau is specifically for conductivity
# a = geta_samples_dst(M, tau_a)[0]
# np.save('conductivity.npy',a)
a = np.load('conductivity.npy')
D = build_D(M,a) # DtN Operator
D_pinv = np.linalg.pinv(D) # NtD Operator

# Trunk Grid
# trunk inputs: border coordinates
# Grid
top_x = np.linspace(0,1,M)
top_x = top_x[0:-1]
right_x = np.ones((M-1,))
bottom_x = np.linspace(1,0,M)
bottom_x = bottom_x[0:-1]
left_x = np.zeros((M-1,))
x = np.concatenate((top_x,right_x,bottom_x,left_x))
x = x.reshape(4*M-4,1)
top_y = np.ones((M-1,))
right_y = np.linspace(1,0,M)
right_y = right_y[0:-1]
bottom_y = np.zeros((M-1,))
left_y = np.linspace(0,1,M)
left_y = left_y[0:-1]
y = np.concatenate((top_y,right_y,bottom_y,left_y))
y = y.reshape(4*M-4,1)
x_trunk = np.concatenate((x,y),1).astype("float32")

def train_model(num_iter,M,N_samples,omega,z=None,display_every=10):
    # Generate DBC data
    h = 1/(M-1)
    num_pts = 4*M-4
    nbc_data = getNBCsamples_MeanShift(num_pts, omega, N_samples=N_samples, z=z)
    nbc_data = nbc_data.astype("float32")
    # corresponding Darcy solutions
    dbc_list = []
    for i in range(N_samples):
        dbc = D_pinv @ nbc_data[i]
        dbc_list.append(dbc - dbc[0])
    dbc_data = np.stack(dbc_list).astype("float32")
    
    # split
    ntrain = int(0.8*N_samples)
    ntest = N_samples - ntrain
    # Cut off corners of DBC to match grid of nbc
    nbc_train, nbc_test = np.split(nbc_data,   [ntrain])
    dbc_train, dbc_test = np.split(dbc_data,   [ntrain])
    
    # flatten for branch: shape (n, M*M)
    v_train = nbc_train
    v_test   = nbc_test
    y_train = dbc_train
    y_test   = dbc_test
    
    # Use a Cartesian‐product dataset with DeepONetCartesianProd
    data = dde.data.TripleCartesianProd(
        X_train=(v_train, x_trunk),
        y_train=y_train,
        X_test =(v_test,  x_trunk),
        y_test =y_test,
    )

    # Build the DeepONet for Cartesian‐product data
    p = 200 # Number of DeepONet basis 
    net = DeepONetCartesianProd(  
        layer_sizes_branch=[4*M-4, 512, 512, 512, p],
        layer_sizes_trunk =[    2, 128, 128, 128, p],
        activation="relu",
        kernel_initializer="Glorot normal",
    )
    
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss ="mean l2 relative error", metrics=["l2 relative error"])
    # model.compile("adam", lr=1e-3, loss ="mse", metrics=["mse"])
    losshistory, train_state  = model.train(iterations=num_iter, display_every=display_every)
    return model, losshistory, train_state

## Testing Distributions
def sample_from_custom_pdf(num_samples, min_x, max_x, num_grid=10001):
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
    x = np.linspace(min_x, max_x, num_grid)
    # Compute unnormalized PDF
    pdf = np.sin(x-np.pi/2)+1
    # Approximate CDF via the trapezoidal rule
    dx = x[1] - x[0]
    cdf = np.cumsum(pdf) * dx
    cdf = cdf / cdf[-1]     # normalize so cdf[-1] == 1
    cdf[0] = 0.0            # enforce CDF(–1) = 0 exactly

    # Remove any repeated CDF values for stable interpolation
    cdf_u, unique_idx = np.unique(cdf, return_index=True)
    x_u = x[unique_idx]

    # Draw uniforms and invert CDF by interpolation
    u = np.random.rand(num_samples)
    samples = np.interp(u, cdf_u, x_u)
    return samples

K = 64
alphak = 1.5
tauk = sample_from_custom_pdf(K,0,50)
sigmak = tauk**(alphak - 1/2)
omegak = sample_from_custom_pdf(K,-2*np.pi,2*np.pi)

## Compute second moment and sqrt eigenvalues for each $\nu'_k$
# Get all (i,j) pairs
N_basis = 4*M-4
i = np.arange(1, N_basis+1)
x = np.linspace(0,1,1000)
mean_normk = np.sum(10*np.sin(omegak[None,:]*(x[:,None]-0.5)),axis=0) # ||mean||^2=\int{mean}dx
m2k = mean_normk + np.sum(sigmak[:,np.newaxis]**2/(((i*np.pi)**2+tauk[:,np.newaxis]**2)**alphak),axis=1)
sqrt_eigk = sigmak[:,np.newaxis]/(((i*np.pi)**2+tauk[:,np.newaxis]**2)**(alphak/2))
Lip=1.5
L_op=50
L_0_norm=1

## Objective Function
def part1(M,N_samples,omega,model,z=None):
    # Must use same M as in model training
    
    # Generate DBC data
    h = 1/(M-1)
    num_pts = 4*M-4
    nbc_data = getNBCsamples_MeanShift(num_pts, omega, N_samples=N_samples, z=z)
    nbc_data = nbc_data.astype("float32")
    # corresponding Darcy solutions
    dbc_list = []
    for i in range(N_samples):
        dbc = D_pinv @ nbc_data[i]
        dbc_list.append(dbc - dbc[0])
    dbc_data = np.stack(dbc_list).astype("float32")

    # Use DeepONet to predicted u
    dbc_pred = model.predict((nbc_data,x_trunk))

    # Return average squared 2-norm
    return np.mean(np.linalg.norm(dbc_data - dbc_pred, ord=2,axis=1)**2)

def part2(omega,M,model,tau=3,sigma=None,alpha=2):
    if sigma == None:
        sigma = tau**(alpha - 1/2)
    x = np.linspace(0,1,1000)
    mean_norm = np.sum(10*np.sin(omega*(x-0.5))) # ||mean||^2=\int{mean}dx
    m2 = mean_norm + np.sum((sigma**2/(((i*np.pi)**2+tau**2)**alpha)))    
    return np.sqrt(np.mean((Lip+L_op)**2*(3*(Lip+L_op)**2*(m2+m2k)+12*L_0_norm**2)))

def part3(omega,tau=3,sigma=None,alpha=2):
    if sigma == None:
        sigma = tau**(alpha - 1/2)
    x = np.linspace(0,1,1000)
    difference_mean_norm = np.sum(10*np.sin(omega*(x[:,None]-0.5)) - 10*np.sin(omegak[None,:]*(x[:,None]-0.5)),axis=0) # ||mean - mean_k||^2=\int{mean}dx
        
    sqrt_eig = sigma/(((i*np.pi)**2+tau**2)**(alpha/2)) # Matrix with \sqrt{\lambda_{i,j}}
    return np.sqrt(np.mean(difference_mean_norm + np.sum((sqrt_eig - sqrt_eigk)**2)))

def obj_fun(omega,model,M,N_samples,tau=3,sigma=None,alpha=2,z=None):
    if sigma == None:
        sigma = tau**(alpha - 1/2)
    
    return part1(M,N_samples,omega,model,z=z)+part2(omega,M,model,tau=tau,sigma=sigma,alpha=alpha)*part3(omega,tau=tau,sigma=sigma,alpha=alpha)

def OOD_denominator(N_samples,z=None):
    # E E [y]
    den = np.zeros(K)
    
    if z is None:
        z = np.random.randn(N_samples, N_basis)
        
    for k in range(K):
        # Generate DBC data
        h = 1/(M-1)
        num_pts = 4*M-4
        nbc_data = getNBCsamples_MeanShift(num_pts, omegak[k], tau=tauk[k], N_samples=N_samples, z=z)
        nbc_data = nbc_data.astype("float32")
        # Compute true DBCs
        dbc_list = []
        for i in range(N_samples):
            dbc = D_pinv @ nbc_data[i]
            dbc_list.append(dbc - dbc[0])
        dbc_data = np.stack(dbc_list).astype("float32")
    
        # Return average squared 2-norm
        den[k]=np.mean(np.linalg.norm(dbc_data, ord=2,axis=1)**2)
    return np.mean(den)
OOD_den = OOD_denominator(1000)

def computeOODerror(model):
    OODerrors = np.zeros(K)
    den = np.zeros(K)
    for k in range(K):
        # Generate DBC data
        h = 1/(M-1)
        num_pts = 4*M-4
        nbc_data = getNBCsamples_MeanShift(num_pts, omegak[k], tau=tauk[k], N_samples=N_samples_train, z=z_train)
        nbc_data = nbc_data.astype("float32")
        # Compute true DBCs
        dbc_list = []
        for i in range(N_samples_train):
            dbc = D_pinv @ nbc_data[i]
            dbc_list.append(dbc - dbc[0])
        dbc_data = np.stack(dbc_list).astype("float32")
    
        # Use DeepONet to predicted DBC
        dbc_pred = model.predict((nbc_data,x_trunk))
    
        # Return average squared 2-norm
        OODerrors[k]=np.mean(np.linalg.norm(dbc_data - dbc_pred, ord=2,axis=1)**2)
        den[k]=np.mean(np.linalg.norm(dbc_data, ord=2,axis=1)**2)
    return np.mean(OODerrors)/np.mean(den)

num_iter_AMA = 10
num_iter_train = 2000
N_samples_train = 500
N_samples_min = 500
num_trials = 80
trial_results = []
for trial in range(num_trials):
    print(f'Working on Trial {trial}')
    z_train = np.random.randn(N_samples_train, N_basis)
    z_min = np.random.randn(N_samples_min, N_basis)
    omega = 8*np.pi # Initial omega
    AMA_loss = 1e+99
    results = []
    for i in range(num_iter_AMA):
        # Part 1
        print("TRAINING")
        model,_,_ = train_model(num_iter_train,M,N_samples_train,omega,z=z_train,display_every=100)
        temp_loss = obj_fun(omega, model, M, N_samples_train, z=z_train)/OOD_den
        attempt = 0
        while temp_loss > AMA_loss and attempt < 10:
            print(f"Hmm, loss {temp_loss} is not smaller than previous loss {AMA_loss}. We'll keep training.")
            # TRAIN AGAIN
            losshistory, train_state = model.train(
                 iterations=num_iter_train,
                 display_every=500,
                 disregard_previous_best=True,
             )
            temp_loss = obj_fun(omega, model, M, N_samples_train, z=z_train)/OOD_den
            attempt += 1
        AMA_loss = temp_loss

        OODerror = computeOODerror(model)

        print(f"Iteration {i+0.5} | Loss: {AMA_loss} | Relative OOD Error: {OODerror} | w: {omega}")
        results.append([i+0.5,AMA_loss,OODerror,omega])
        df = pd.DataFrame(results, columns = ['Iteration','AMA Loss','Relative OOD Error','w'])

        # Part 2
        print("MINIMIZING")
        tol = 1e-7
        def obj_w(omega):
            return obj_fun(omega, model, M, N_samples_min, z=z_min)/OOD_den
        res = differential_evolution(obj_w, bounds=[(-9*np.pi, 9*np.pi)], tol=tol)
        temp_omega = res.x[0]
        temp_loss = res.fun

        attempt=0
        while temp_loss >= AMA_loss and attempt < 10:
            print(f"Hmm, loss {temp_loss} is not less than previous loss {AMA_loss}. Let's use smaller tolerance for minimizing.")
            tol = tol/2
            res = differential_evolution(obj_w, bounds=[(-9*np.pi, 9*np.pi)], tol=tol)
            temp_omega  = res.x[0]
            temp_loss = res.fun
            attempt += 1

        omega = temp_omega
        AMA_loss = temp_loss
        print(f"Iteration {i+1} | Loss: {AMA_loss} | OOD Error: NA | w: {omega} ")
        results.append([i+1,AMA_loss,np.nan,omega])
        df = pd.DataFrame(results, columns = ['Iteration','AMA Loss','Relative OOD Error','w'])
    trial_results.append(df)
    with open('NtD_results.pkl', 'wb') as f:
        pickle.dump(trial_results, f)

