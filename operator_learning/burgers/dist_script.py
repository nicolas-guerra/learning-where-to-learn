from burgers import *
import scipy.io
import numpy as np
import random
import os
import csv
from numpy import genfromtxt
import gc

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
n = a_test.shape[1]

optimal_params = genfromtxt('AMALogs/optimal_parameters.csv', delimiter=',')[1]
alpha = optimal_params[0]; tau = optimal_params[1]; sigma = optimal_params[2]
GP_train = MaternGP(n,alpha=alpha,tau=tau,sigma=sigma,device=device)

# Create log directory if it doesn't exist
os.makedirs("DistLogs", exist_ok=True)

num_trials = 10
for i in range(num_trials):

    # Initialize CSV file with header
    with open(f"DistLogs/dist_trial{i}_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Relative_OOD_Error"])
    
    n_total = 1000
    n_init = 10
    n_rounds = 20 
    # n_select = (n_total-n_init)//n_rounds # Each round, we add 50 points to eventually get to n_total=600.
    n_select_first = 40
    n_select = 50 # Except for the first round which will be 40.
    
    # Initialize burger class with N
    burg = Burgers(n_init,n,device)
    # Draw n_init samples from distribution
    a_train = GP_train.sample(n_init)
    u_train = burg.burgers(a_train)
    # Train model
    model, _, _ = burg.train_model(a_train, u_train)
    error = burg.relativeOODerror(model, GP_valid)
    print(f'N: {n_init} | Relative OOD Error: {error.item()}')
    with open(f"DistLogs/dist_trial{i}_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([n_init, error.item()])
    
    for round in range(n_rounds):
        print(f"\nRound {round+1}/{n_rounds}")
    
        # Sample from training distribution
        if round == 0:
            N = n_init + n_select_first
        else:
            N = n_init + n_select_first + round*n_select
        # Initialize burger class with N
        burg = Burgers(N,n,device)
        # Draw n_init samples from distribution
        a_train = GP_train.sample(N)
        u_train = burg.burgers(a_train)

        # Delete current model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train model
        model, _, _ = burg.train_model(a_train, u_train)
        error = burg.relativeOODerror(model, GP_valid)
        print(f'N: {N} | Relative OOD Error: {error.item()}')
        with open(f"DistLogs/dist_trial{i}_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([N, error.item()])