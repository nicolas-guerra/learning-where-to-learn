from rte import *
import math
import matplotlib.pyplot as plt
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
from time import time
import warnings
warnings.filterwarnings('ignore', message='.*rcond.*')
import csv
import numpy as np
import random
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description='Run particle optimization')
parser.add_argument('--ep_index', type=int, required=True, help='Epsilon index')
parser.add_argument('--N', type=int, required=True, help='Number of particles')
args = parser.parse_args()

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = 1.0
dr = 0.02
dtheta = 2 * math.pi / 12
ep_index = args.ep_index 
N = args.N  # Number of particles
Ntrunc = 20
myRTE = RTESolver(L, dr, dtheta, ep_index, device)
num_trials = 10

# Fixed Test Distributions, 3 distributions so K=3.
GP1 = MaternGP(Ntrunc, dim=2, alpha=2, tau=1, sigma=1, device=device) # Define to get Ntrunc^2 eigenvalues of matern kernel
GP2 = MaternGP(Ntrunc, dim=2, alpha=2, tau=1, sigma=2, device=device)
GP3 = MaternGP(Ntrunc, dim=2, alpha=2, tau=1, sigma=3, device=device)
GP_mean = MaternGP(myRTE.Nr, dim=2, alpha=2, tau=1, device=device) # Define to sample an NrxNr matrix as mean

# Multi-trial
for trial_j in range(num_trials):
    print('Generating test distributions.')
    c_test1 = GP1.sqrt_eig[None,:,:]*torch.randn(N,Ntrunc,Ntrunc,device=device)
    c_test2 = GP2.sqrt_eig[None,:,:]*torch.randn(N,Ntrunc,Ntrunc,device=device)
    c_test3 = GP3.sqrt_eig[None,:,:]*torch.randn(N,Ntrunc,Ntrunc,device=device)
    GP_mean1 = GP_mean.sample(1)
    GP_mean2 = GP_mean.sample(1)
    GP_mean3 = GP_mean.sample(1)
    loga_test1 = myRTE.c_to_loga(c_test1) + GP_mean1 # Add the same function to dist. to have nonzero mean
    loga_test2 = myRTE.c_to_loga(c_test2) + GP_mean2 # ""
    loga_test3 = myRTE.c_to_loga(c_test3) + GP_mean3 # ""
    # loga_test1 = myRTE.c_to_loga(c_test1)
    # loga_test2 = myRTE.c_to_loga(c_test2)
    # loga_test3 = myRTE.c_to_loga(c_test3)
    a_test1 = torch.exp(loga_test1)
    a_test2 = torch.exp(loga_test2)
    a_test3 = torch.exp(loga_test3)
    c_test = torch.stack((c_test1,c_test2,c_test3))
    a_test = torch.stack((a_test1,a_test2,a_test3))
    # Compute true u_test
    u_test = torch.zeros_like(a_test,device=device)
    K = a_test.shape[0]
    for k in range(K):
        u_test[k] = myRTE.compute_u(a_test[k])
        
    print('Generating held-out test distributions for OOD Error computation.')
    c_OOD1 = GP1.sqrt_eig[None,:,:]*torch.randn(N,Ntrunc,Ntrunc,device=device)
    c_OOD2 = GP2.sqrt_eig[None,:,:]*torch.randn(N,Ntrunc,Ntrunc,device=device)
    c_OOD3 = GP3.sqrt_eig[None,:,:]*torch.randn(N,Ntrunc,Ntrunc,device=device)
    loga_OOD1 = myRTE.c_to_loga(c_OOD1) + GP_mean1 # Add the same function to dist. to have nonzero mean
    loga_OOD2 = myRTE.c_to_loga(c_OOD2) + GP_mean2 # ""
    loga_OOD3 = myRTE.c_to_loga(c_OOD3) + GP_mean3 # ""
    # loga_test1 = myRTE.c_to_loga(c_test1)
    # loga_test2 = myRTE.c_to_loga(c_test2)
    # loga_test3 = myRTE.c_to_loga(c_test3)
    a_OOD1 = torch.exp(loga_OOD1)
    a_OOD2 = torch.exp(loga_OOD2)
    a_OOD3 = torch.exp(loga_OOD3)
    c_OOD = torch.stack((c_OOD1,c_OOD2,c_OOD3))
    a_OOD = torch.stack((a_OOD1,a_OOD2,a_OOD3))
    # Compute true u_test
    u_OOD = torch.zeros_like(a_OOD,device=device)
    K = a_OOD.shape[0]
    for k in range(K):
        u_OOD[k] = myRTE.compute_u(a_OOD[k])

    # Compute expected 2nd moment of test distribution
    test_m2s = torch.zeros(K,device=device)
    for k in range(K):
        test_m2s[k] = myRTE.m2(a_test[k])
    mean_test_m2 = torch.mean(test_m2s)

    # Set important parameters and save test distribution
    a_zero_coef = myRTE.c_to_a(torch.zeros(1,Ntrunc,Ntrunc,device=device))
    true_model_zero = myRTE.norm(myRTE.SolveRTE(a_zero_coef)[0])**2
    Lip, R = 1, 1
    myRTE.set_parameters(Lip, R, mean_test_m2, true_model_zero, c_test, a_test, u_test, a_OOD, u_OOD, a_zero_coef)

    # Benchmark: Train on testing distribution and save OOD error (in this case, ID error)
    # I will sample uniformly from {1,2,3} and then samples from \nu_i' and do this N times
    print("Making benchmark data.")
    GPs = {1: GP1, 2: GP2, 3: GP3}
    a_benchmark = torch.zeros(N, myRTE.Nr, myRTE.Nr, device=GP1.device)
    # Sample N times
    for n in range(N):
        # Uniformly choose which distribution to sample from
        i = random.choice([1, 2, 3])
        # Sample from chosen distribution and squeeze the batch dimension
        random_c = GPs[i].sqrt_eig[None,:,:]*torch.randn(1,Ntrunc,Ntrunc,device=device)
        a_benchmark[n] = torch.exp(myRTE.c_to_loga(random_c) + GP_mean.sample(1)).squeeze(0)
    u_benchmark = myRTE.compute_u(a_benchmark)
    print('Training benchmark model.')
    model_benchmark,_,_ = myRTE.train_model(a_benchmark,u_benchmark)
    benchmark = myRTE.computeOODError(model_benchmark)
    relative_benchmark = myRTE.computeRelativeOODError(model_benchmark)

    # Initialize Training Distribution
    # c_train = torch.zeros(N,Ntrunc,Ntrunc,device=device)
    print('Initializing Training Distribution')
    c_train = 0.001 * torch.randn(N, Ntrunc, Ntrunc, device=device)
    c_train = c_train.detach().requires_grad_(True)
    a_train = myRTE.c_to_a(c_train) # Has gradient info wrt c_train_
    a_train_det = a_train.detach().clone() # Doesn't have gradient info. Used for training

    # Compute true u_train
    print('Computing Training Outputs')
    u_train, f_train = myRTE.compute_u(a_train_det, get_soln=True) # No grad info, not needed to compute grad since we do adjoint method
    
    # Train initial model
    print('Training Model with Training Distribution')
    model,_,_ = myRTE.train_model(a_train_det,u_train) # train without gradient info 

    # Main Gradient Descent Loop
    # Create the directory if it doesn't exist
    directory = f'test_training_log_eps{myRTE.epsilon}N{N}'
    os.makedirs(directory, exist_ok=True)
    # Create/initialize the CSV file with headers
    csv_filename = f'{directory}/test_training_log_{trial_j}.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Objective_Function', 'OOD_Error', 'Relative_OOD_Error', 'Benchmark_OOD', 'Relative_Benchmark_OOD', 'Step_Size', 'Time (s)'])

    num_iter = 50 # Max Number of Iterations
    OODerrors = torch.zeros(num_iter+1, device=device)
    relativeOODerrors = torch.zeros(num_iter+1, device=device)
    Fs = torch.zeros(num_iter+1, device=device)
    OODerrors[0] = myRTE.computeOODError(model)
    relativeOODerrors[0] = myRTE.computeRelativeOODError(model)
    Fs[0] = myRTE.F(c_train,u_train,model)
    start = time()
    print(f'Iteration: {0} | Objective Function: {Fs[0]} | OOD Error: {OODerrors[0]} | Relative OOD Error: {relativeOODerrors[0]} | Benchmark: {benchmark} | Relative Benchmark: {relative_benchmark} | Step Size: NA | Time: 0')

    # Save iteration 0 to CSV
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, Fs[0].item(), OODerrors[0].item(), relativeOODerrors[0].item(), benchmark.item(), relative_benchmark.item(), 'NA', 0])

    stepsize = 1e-6 # Note if stepsize is decreased in future iterations, it stays decreased.
    tol = 1e-6 # Tolerance to reach
    print(f'Step Norm Tolerance: {tol}')
    for i in range(1, num_iter+1):
        success = False
        grad = myRTE.compute_gradient(c_train,a_train,f_train,u_train,model) # Use c_train, u_train with grad info
        step = stepsize*grad
        print(f"Norm of Current Step: {torch.sqrt(torch.sum(step**2))}")
        if torch.sqrt(torch.sum(step**2)) < tol:
            print(f"Step norm too small. Script Terminating.")
            break
        temp_c_train = c_train - step
        temp_a_train = myRTE.c_to_a(temp_c_train)
        temp_a_train_det = temp_a_train.detach().clone()
        temp_u_train, temp_f_train = myRTE.compute_u(temp_a_train_det, get_soln=True)
        
        temp_model,_,_ = myRTE.train_model(temp_a_train_det,temp_u_train)
        temp_OODerror = myRTE.computeOODError(temp_model)
        temp_relativeOODerror = myRTE.computeRelativeOODError(temp_model)
        temp_F = myRTE.F(temp_c_train,temp_u_train,temp_model)

        while temp_F > Fs[i-1]:
            print(f'Hmm, loss {temp_F} is worse than {Fs[i-1]}. Decreasing step size from {stepsize} to {stepsize/10} and trying again.')
            stepsize /= 10
            step = stepsize*grad
            print(f"Norm of Current Step: {torch.sqrt(torch.sum(step**2))}")
            if torch.sqrt(torch.sum(step**2)) < tol:
                print(f"Step norm too small. Script Terminating.")
                IMPROVE = False
                break
            temp_c_train = c_train - step
            temp_a_train = myRTE.c_to_a(temp_c_train)
            temp_a_train_det = temp_a_train.detach().clone()
            temp_u_train, temp_f_train = myRTE.compute_u(temp_a_train_det, get_soln=True)
            temp_model,_,_ = myRTE.train_model(temp_a_train_det,temp_u_train)
            temp_OODerror = myRTE.computeOODError(temp_model)
            temp_relativeOODerror = myRTE.computeRelativeOODError(temp_model)
            temp_F = myRTE.F(temp_c_train,temp_u_train,temp_model)

        # Save new train distribution
        c_train = temp_c_train.detach().clone().requires_grad_(True)
        # a_train = temp_a_train.clone()
        # a_train_det = temp_a_train_det.clone()
        a_train = myRTE.c_to_a(c_train)
        a_train_det = a_train.detach().clone()
        u_train = temp_u_train.clone()
        f_train = temp_f_train.clone()
        model = temp_model
        OODerrors[i] = temp_OODerror
        relativeOODerrors[i] = temp_relativeOODerror
        Fs[i] = temp_F
        current_time = time() - start
        print(f'Iteration: {i} | Objective Function: {Fs[i]} | OOD Error: {OODerrors[i]} | Relative OOD Error: {relativeOODerrors[i]} | Benchmark: {benchmark} | Relative Benchmark: {relative_benchmark} | Step Size: {stepsize} | Time: {current_time}')

        # Save current iteration to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i, Fs[i].item(), OODerrors[i].item(), relativeOODerrors[i].item(), benchmark.item(), relative_benchmark.item(), stepsize, current_time])

        print(f'Improvement Made: {torch.round(torch.abs((temp_F - Fs[i-1])/Fs[i-1])*100,decimals=3)}% decrease')