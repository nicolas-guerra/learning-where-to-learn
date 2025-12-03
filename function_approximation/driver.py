import torch
import numpy as np
import os
from tqdm import tqdm
from util.utilities_module import CosineAnnealingLR, get_emp_mean, get_emp_cov
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
save_pref = "sobol_compare_iclr"
USE_MEAN = True
SPLIT_DATA = not True
plot_folder = "./results/" + save_pref + "/"
os.makedirs(plot_folder, exist_ok=True)


torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)


# TODO
# Import config
import config_sobolg as config

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
# truth = rkhs_func(d, J=1000, crange=4, sigma=sigma, kernel=gaussian_kernel)
truth = sobol_func
# truth = friedmann1_func
# truth = friedmann2_func

kernel = lambda x, y: gaussian_kernel(x, y, sigma=sigma) # faster
# kernel = lambda x, y: laplace_kernel(x, y, sigma=sigma) # slower


# Generate random measure over test distributions
means = Q_sigma*torch.randn(J, d)
W = torch.distributions.wishart.Wishart(torch.Tensor([d + 1]), scale_tril=(Q_gamma * torch.eye(d)))
covs = W.sample((J,)).squeeze(dim=1)
mu_list = []
data_test = []
data_test_test = []
for j in range(J):
    # Inputs
    mu = torch.distributions.multivariate_normal.MultivariateNormal(means[j, ...], 
                                                                    covariance_matrix=covs[j, ...])
    mu_list.append(mu)
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

errors = torch.zeros(num_loops, steps, 2)
if USE_MEAN:
    parameters = torch.zeros(num_loops, steps, d + (d*(d + 1) // 2))
    design_class = DesignDistribution_WithMean
    design_class_init = (d, my_mean*torch.ones(d))
else:
    parameters = torch.zeros(num_loops, steps, d*(d + 1) // 2)
    design_class = DesignDistribution
    design_class_init = (my_mean*torch.ones(d),)

x_fevals = torch.zeros(num_loops, steps + 1)

for loop in tqdm(range(num_loops)):
    print(f"MC Loop {loop}/{num_loops}")
    
    design = design_class(*design_class_init, device=device)   # initialize design class

    for step in tqdm(range(steps)):
        # Current parameter
        theta = design.theta
        
        # Sample training data from current design, train model, and solve adjoint for current design
        eps = reg_scheduler(step)
        N_step = N_scheduler(steps - step).__floor__()
        model, X, Y = model_update(N_step, design, truth, eps, data_test, kernel, device, return_data=True, split_data=SPLIT_DATA)
    
        # Gradient update
        my_grad = (model(X) - Y) * model.adjoint_values
        my_grad = my_grad.unsqueeze(-1) * design.parametric_score(X, theta)
        theta = theta - scheduler(step) * torch.mean(my_grad, dim=0)
        
        # Set new parameter
        design.set_params(theta)
        
        # Log
        loss = meta_loss(model, data_test, device=device)
        loss_unseen = meta_loss(model, data_test_test, device=device)
        errors[loop, step, 0] = loss
        errors[loop, step, 1] = loss_unseen
        parameters[loop, step, ...] = theta
        x_fevals[loop, step + 1] = x_fevals[loop, step] + N_step
        print(f'Step [{step+1}/{steps}], MetaLoss (seen): {loss}, MetaLoss (unseen): {loss_unseen}')

# Save looped results
print(f'Optimal Theta: {theta.detach().cpu()}')
np.save(plot_folder + "errors", errors.cpu().numpy())
np.save(plot_folder + "parameters", parameters.detach().cpu().numpy())
np.save(plot_folder + "x_fevals", x_fevals.cpu().numpy())


# Fixed distribution training for KRR
def loop_krr(design, sample_size_list, eps=eps_initial, device=device, truth=truth, data_test=data_test, data_test_test=data_test_test, kernel=kernel, meta_loss=meta_loss):
    """
    Fix a training distribution, and loop through training sample sizes
    """
    eps_init = eps
    errors = []
    for N in sample_size_list:
        # Build model
        eps = eps_init / N      # Scale regularization with N in Bayesian way
        try:
            model = model_update(N, design, truth, eps, data_test, kernel, device, split_data=False)
        except:
            model = model_update(N, design, truth, eps, data_test, kernel, device="cpu", split_data=False)

        # Log
        loss = meta_loss(model, data_test_test, device=device)
        errors.append(loss)
        print(f'Sample Size [{N}/{sample_size_list[-1]}], MetaLoss (unseen): {loss}')
        
    return torch.tensor(errors)


# coreset loop
def loop_coreset(batch, my_core, sample_size_list, eps=eps_initial, device=device, truth=truth, data_test=data_test, data_test_test=data_test_test, kernel=kernel, meta_loss=meta_loss):
    eps_init = eps
    errors = []
    Npool = my_core.Npool
    sample_size_list = sample_size_list[sample_size_list < Npool]
    
    count = 0
    for _ in range(sample_size_list[-1]//batch - my_core.init_select):
        _ = my_core.select(batch)
        pts = design_ncore.get_points()
        N = pts.shape[0]
        if N >= sample_size_list[count]:
            count += 1

            # Build model
            eps = eps_init / N      # Scale regularization with N in Bayesian way
            try:
                model = model_update_coreset(pts, truth, eps, data_test, kernel, device, split_data=False)
            except:
                # my_core.device = torch.device("cpu")
                model = model_update_coreset(pts.to("cpu"), truth, eps, data_test, kernel, device="cpu", split_data=False)
    
            # Log
            loss = meta_loss(model, data_test_test, device=device)
            errors.append(loss)
            print(f'Sample Size [{N}/{sample_size_list[-1]}], MetaLoss (unseen): {loss}')
        
        if count >= sample_size_list.shape[0]:
            break
        
    return torch.tensor(errors)


# Build comparison distributions
mu_emp_list = []
means_emp = torch.zeros(J,d)
covs_emp = torch.zeros(J,d,d)
for j in range(J):
    emean = get_emp_mean(data_test[j][0])
    ecov = get_emp_cov(data_test[j][0])
    means_emp[j, ...] = emean
    covs_emp[j, ...] = ecov
    mu_emp = torch.distributions.multivariate_normal.MultivariateNormal(emean, 
                                                                    covariance_matrix=ecov)
    mu_emp_list.append(mu_emp)


design_static = design_class(*design_class_init, device=device)

design_seq = design_class(*design_class_init, device=device)
design_seq.set_params(theta)

design_unif = UniformDistribution(d, device=device)

design_gmm = FiniteMixture(mu_emp_list, device=device)

design_bary = GaussianBarycenter(means_emp, covs_emp, device=device)

# coreset
batch=1
pool = [x[0] for x in data_test]
pool = torch.cat(pool, 0)


# Save covs
cov_seq = design_seq.distribution.covariance_matrix.cpu().numpy()
cov_mix = get_emp_cov(torch.cat([data_test[j][0] for j in range(J)], 0)).cpu().numpy()
cov_bary = design_bary.bcov.cpu().numpy()
cov_to_save = np.zeros((3, d, d))
for i, cov in enumerate([cov_seq, cov_mix, cov_bary]):
    cov_to_save[i, ...] = cov
np.save(plot_folder + "cov_emp", cov_to_save)


# Setup sample complexity loop
sample_size_list = 2**torch.arange(4,15)
np.save(plot_folder + "sample_size_list", sample_size_list.cpu().numpy())

errors_static_loops = []
errors_seq_loops = []
errors_unif_loops = []
errors_gmm_loops = []
errors_bary_loops = []
errors_ncore_loops = []
for _ in tqdm(range(num_loops)):

    # Initial
    print("Initial static")
    errors_static_loops.append(loop_krr(design_static, sample_size_list))
    
    # Final
    print("Final adaptive")
    errors_seq_loops.append(loop_krr(design_seq, sample_size_list))
    
    # Uniform sampling
    print("Uniform sampling")
    errors_unif_loops.append(loop_krr(design_unif, sample_size_list))
    
    # Gaussian mixture
    print("Gaussian mixture")
    errors_gmm_loops.append(loop_krr(design_gmm, sample_size_list))
    
    # Barycenter
    print("Barycenter")
    errors_bary_loops.append(loop_krr(design_bary, sample_size_list))

    # Nonadaptive coreset
    print("Nonadaptive Coreset")
    design_ncore = NonadaptiveCoreSet(kernel=kernel, pool=pool, device=device)
    errors_ncore_loops.append(loop_coreset(batch, design_ncore, sample_size_list))

# Save MC loops
errors_static_loops = torch.stack(errors_static_loops)
errors_seq_loops = torch.stack(errors_seq_loops)
errors_unif_loops = torch.stack(errors_unif_loops)
errors_gmm_loops = torch.stack(errors_gmm_loops)
errors_bary_loops = torch.stack(errors_bary_loops)
errors_ncore_loops = torch.stack(errors_ncore_loops)

np.save(plot_folder + "errors_static_loops", errors_static_loops.cpu().numpy())
np.save(plot_folder + "errors_seq_loops", errors_seq_loops.cpu().numpy())
np.save(plot_folder + "errors_unif_loops", errors_unif_loops.cpu().numpy())
np.save(plot_folder + "errors_gmm_loops", errors_gmm_loops.cpu().numpy())
np.save(plot_folder + "errors_bary_loops", errors_bary_loops.cpu().numpy())
np.save(plot_folder + "errors_ncore_loops", errors_ncore_loops.cpu().numpy())
