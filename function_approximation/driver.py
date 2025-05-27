import torch
import numpy as np
import os
from tqdm import tqdm
from util.utilities_module import CosineAnnealingLR
from kernels import *
from testfunctions import *
from bilevel import *

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

# TODO
# USER INPUT
save_pref = "f2_d5_eval"
plot_folder = "./results/" + save_pref + "/"
os.makedirs(plot_folder, exist_ok=True)


torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)


# TODO
# Import config
import config_f2 as config

FLOAT64_FLAG = config.FLOAT64_FLAG
d = config.d
sigma = config.sigma
J = config.J
Q_sigma = config.Q_sigma
Q_gamma = config.Q_gamma
N_final = config.N_final
N_initial = config.N_initial
N_test_total = config.N_test_total
N_val = config.N_val
my_mean = config.my_mean
steps = config.steps
lr_initial = config.lr_initial
eps_initial = config.eps_initial
eps_final = config.eps_final
num_loops = config.num_loops

if FLOAT64_FLAG:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)

# TODO
# Set kernel and ground truth function
# truth = sobol_func
# truth = ishigimi_func
# truth = friedmann1_func
truth = friedmann2_func


kernel = lambda x, y: gaussian_kernel(x, y, sigma=sigma) # faster
# kernel = lambda x, y: laplace_kernel(x, y, sigma=sigma) # slower


# Generate random measure over test distributions
means = Q_sigma*torch.randn(J, d)
W = torch.distributions.wishart.Wishart(torch.Tensor([d + 1]), scale_tril=(Q_gamma * torch.eye(d)))
covs = W.sample((J,)).squeeze()
data_test = []
data_test_test = []
for j in range(J):
    # Inputs
    mu = torch.distributions.multivariate_normal.MultivariateNormal(means[j, ...], 
                                                                    covariance_matrix=covs[j, ...])
    V = mu.sample((N_test_total,))
    
    # Labeled outputs
    Y = truth(V)
    
    # Store
    data_test.append((V[:N_val,...], Y[:N_val,...]))
    data_test_test.append((V[N_val:,...], Y[N_val:,...]))


# Gradient descent
scheduler = CosineAnnealingLR(steps, lr_initial)
reg_scheduler = CosineAnnealingLR(steps, eps_initial, eps_final)
N_scheduler = CosineAnnealingLR(steps, N_final, N_initial)

errors = torch.zeros(num_loops, steps)
parameters = torch.zeros(num_loops, steps, d*(d + 1) // 2)
x_fevals = torch.zeros(num_loops, steps + 1)

for loop in tqdm(range(num_loops)):
    print(f"MC Loop {loop}/{num_loops}")
    
    design = DesignDistribution(my_mean*torch.ones(d), device=device)   # initialize design class
    
    for step in tqdm(range(steps)):
        # Current parameter
        theta = design.theta
        
        # Sample training data from current design, Build model and solve adjoint for current design
        eps = reg_scheduler(step)
        N_step = N_scheduler(steps - step).__floor__()
        model, X, Y = model_update(N_step, design, truth, eps, data_test, kernel, device, return_data=True)
    
        # Gradient update
        my_grad = (model.model_values - Y) * model.adjoint_values
        my_grad = my_grad.unsqueeze(-1) * design.parametric_score(X, theta)
        theta = theta - scheduler(step) * torch.mean(my_grad, dim=0)
        
        # Set new parameter
        design.set_params(theta)
        
        # Log
        loss = meta_loss(model, data_test, device=device)
        errors[loop, step] = loss
        parameters[loop, step, ...] = theta
        x_fevals[loop, step + 1] = x_fevals[loop, step] + N_step
        print(f'Step [{step+1}/{steps}], MetaLoss: {loss}')

# Save looped results
print(f'Optimal Theta: {theta.detach().cpu()}')
np.save(plot_folder + "errors", errors.cpu().numpy())
np.save(plot_folder + "parameters", parameters.detach().cpu().numpy())
np.save(plot_folder + "x_fevals", x_fevals.cpu().numpy())


# Fixed distribution training for KRR
def loop_krr(sample_size_list, eps, my_mean=my_mean, d=d, device=device, truth=truth, data_test=data_test_test, kernel=kernel, meta_loss=meta_loss, theta=None):
    """
    Fix a training distribution, and loop through training sample sizes
    """
    design = DesignDistribution(my_mean*torch.ones(d), device=device)
    if theta is not None:
        design.set_params(theta)
    eps_init = eps
    errors = []
    for N in sample_size_list:
        # Build model
        eps = eps_init / N      # Scale regularization with N in Bayesian way
        try:
            model = model_update(N, design, truth, eps, data_test, kernel, device)
        except:
            model = model_update(N, design, truth, eps, data_test, kernel, device="cpu")

        # Log
        loss = meta_loss(model, data_test, device=device)
        errors.append(loss)
        print(f'Sample Size [{N}/{sample_size_list[-1]}], MetaLoss: {loss}')
        
    return torch.tensor(errors)

sample_size_list = 2**torch.arange(4,15)
np.save(plot_folder + "sample_size_list", sample_size_list.cpu().numpy())

errors_static_loops = []
errors_seq_loops = []
for _ in tqdm(range(num_loops)):

    # Initial
    print("Initial static")
    errors_static = loop_krr(sample_size_list, eps_initial, data_test=data_test_test)
    
    # Final
    print("Final adaptive")
    errors_seq = loop_krr(sample_size_list, eps_initial, data_test=data_test_test, theta=theta)
    
    # Store
    errors_static_loops.append(errors_static)
    errors_seq_loops.append(errors_seq)

# Save MC loops
errors_static_loops = torch.stack(errors_static_loops)
errors_seq_loops = torch.stack(errors_seq_loops)
np.save(plot_folder + "errors_static_loops", errors_static_loops.cpu().numpy())
np.save(plot_folder + "errors_seq_loops", errors_seq_loops.cpu().numpy())
