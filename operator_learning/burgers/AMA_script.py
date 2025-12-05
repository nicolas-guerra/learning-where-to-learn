from burgers import *
import scipy.io
from scipy.optimize import minimize
import numpy as np
import random
import os
import csv

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read in data
data = scipy.io.loadmat('burgers_data_R10.mat')
num_data = data['a'].shape[0]
num_test = int(0.9 * num_data)  
indices = np.random.permutation(num_data)

# Split given data between 90% test data/pool data and 10% validation data to compute OOD error
a_test = torch.tensor(data['a'][indices[:num_test]], dtype=torch.float32, device=device) 
u_test = torch.tensor(data['u'][indices[:num_test]], dtype=torch.float32, device=device)
a_valid = torch.tensor(data['a'][indices[num_test:]], dtype=torch.float32, device=device)
u_valid = torch.tensor(data['u'][indices[num_test:]], dtype=torch.float32, device=device)
GP_test = TestDistribution(a_test,u_test,device)
GP_valid = TestDistribution(a_valid,u_valid,device)

M = a_test.shape[0] # Number of samples in a_test to find optimal training distribution 
n = a_test.shape[1] # Number of discretization points per function

# Initial Training Distribution
# alpha = 3; tau = 2; sigma = 2**1.5
alpha = 3; tau = 1.4; sigma = 5
burg = Burgers(M,n,device)
GP_train = MaternGP(n, alpha=alpha, tau=tau, sigma=sigma, device=device)
z = torch.randn(M,2*(GP_train.size//2), device=GP_train.device)
a_train = GP_train.sample(M,z=z)
u_train = burg.burgers(a_train)

# Train Initial Model
model,_,_ = burg.train_model(a_train, u_train)

# Create AMALogs directory if it doesn't exist
os.makedirs("AMALogs", exist_ok=True)

# Initialize log file with headers
with open("AMALogs/log.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Alpha", "Tau", "Sigma", "AMA_Loss", "Relative_OOD_Error"])

# Compute Initial Error
OODerror = burg.relativeOODerror(model,GP_valid)
obj = burg.F(alpha, tau, sigma, model, GP_test, z=z)
print(f"Iteration: {0} | Alpha: {alpha:.4f} | Tau: {tau:.4f} | Sigma: {sigma:.4f} | AMA Loss: {obj.item():.4f} | Relative OOD Error: {OODerror.item():.4f}")

# Log initial iteration
with open("AMALogs/log.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([0, alpha, tau, sigma, obj.item(), OODerror.item()])

num_iter = 10 # Number of AMA iterations
for i in range(num_iter):
    def objective(params):
        """Objective function for scipy"""
        alpha, tau, sigma = params
        loss = burg.F(alpha, tau, sigma, model, GP_test, z=z)
        return loss.item()  # Convert to scalar
    
    def gradient(params):
        """Gradient function for scipy"""
        alpha, tau, sigma = params
        grads = burg.grad_F(alpha, tau, sigma, model, GP_test, z=z)
        return grads.detach().cpu().numpy()  # Convert to numpy array
    
    # Initial parameter values
    x0 = np.array([alpha, tau, sigma])
    print("UPDATING DISTRIBUTION PARAMETERS WITH 1 ITERATION OF L-BFGS-B.")
    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',  # 'L-BFGS-B', 'BFGS', 'CG', etc.
        jac=gradient, 
        options={'disp': True, 'maxiter': 1}
    )
    
    # Update Distribution Parameters
    alpha_prev, tau_prev, sigma_prev = alpha, tau, sigma
    obj_prev = obj
    alpha, tau, sigma = result.x
    obj = result.fun
    # Train model on new distribution
    GP_train = MaternGP(n, alpha=alpha, tau=tau, sigma=sigma, device=device)
    a_train = GP_train.sample(M,z=z)
    u_train = burg.burgers(a_train)
    # Compute AMA loss and OOD error of updated model
    model,_,_ = burg.train_model(a_train, u_train)
    obj = burg.F(alpha, tau, sigma, model, GP_test, z=z)
    OODerror = burg.relativeOODerror(model,GP_valid)
    if obj > obj_prev:
        print('No further progress possible with current training infrastructure. Script terminating.')
        alpha, tau, sigma = alpha_prev, tau_prev, sigma_prev
        print(f'Optimized Distribution Parameters are Alpha: {alpha:.4f}, Tau: {tau:.4f}, Sigma: {sigma:.4f}')
        # Save optimal parameters
        with open("AMALogs/optimal_parameters.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Alpha", "Tau", "Sigma"])
            writer.writerow([alpha, tau, sigma])
        break
    print(f"Iteration: {i+1} | Alpha: {alpha:.4f} | Tau: {tau:.4f} | Sigma: {sigma:.4f} | AMA Loss: {obj.item():.4f} | Relative OOD Error: {OODerror.item():.4f}")
    
    # Log this iteration
    with open("AMALogs/log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([i+1, alpha, tau, sigma, obj.item(), OODerror.item()])
    
    if i+1 == num_iter:
        print('Max number of iterations reached. Script terminating.')
        print(f'Optimized Distribution Parameters are Alpha: {alpha:.4f}, Tau: {tau:.4f}, Sigma: {sigma:.4f}')
        # Save optimal parameters
        with open("AMALogs/optimal_parameters.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Alpha", "Tau", "Sigma"])
            writer.writerow([alpha, tau, sigma])