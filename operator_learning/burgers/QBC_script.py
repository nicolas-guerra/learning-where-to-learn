from activelearning import *
import scipy.io
import numpy as np
import random
import os
import csv
import gc

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = scipy.io.loadmat('burgers_data_R10.mat')
num_data = data['a'].shape[0]
num_pool = int(0.9 * num_data)  
indices = np.random.permutation(num_data)

# Split given data between 90% test data/pool data and 10% validation data to compute OOD error
a_pool = torch.tensor(data['a'][indices[:num_pool]], dtype=torch.float32, device=device) 
u_pool = torch.tensor(data['u'][indices[:num_pool]], dtype=torch.float32, device=device)
a_valid = torch.tensor(data['a'][indices[num_pool:]], dtype=torch.float32, device=device)
u_valid = torch.tensor(data['u'][indices[num_pool:]], dtype=torch.float32, device=device)

# Create log directory if it doesn't exist
os.makedirs("ALLogs", exist_ok=True)

num_trials = 10
for i in range(num_trials):
    # Initialize CSV file with header
    with open(f"ALLogs/QBC_trial{i}_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Relative_OOD_Error"])
    
    al = ActiveLearning(a_valid, u_valid, device)
    n_total = 1000
    n_init = 10
    n_rounds = 20 
    # n_select = (n_total-n_init)//n_rounds # Each round, we add 50 points to eventually get to n_total=600.
    n_select_first = 40
    n_select = 50 # Except for the first round which will be 40.
    
    # Start with random initial points
    all_indices = torch.randperm(a_pool.shape[0])
    train_indices = all_indices[:n_init]
    # Initial training data
    a_train = a_pool[train_indices]
    u_train = u_pool[train_indices]
    # Train initial ensemble
    print("Training initial ensemble...")
    ensemble = al.train_ensemble(a_train, u_train)
    # Compute Relative OOD Error - Save the first ensemble error
    error = al.relativeOODerror(ensemble[0])
    
    print(f'N: {n_init} | Relative OOD Error: {error.item()}')
    with open(f"ALLogs/QBC_trial{i}_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([n_init, error.item()])
    
    for round in range(n_rounds):
        print(f"\nRound {round+1}/{n_rounds}")
        
        # Create pool (exclude already selected points)
        mask = torch.ones(a_pool.shape[0], dtype=torch.bool)
        mask[train_indices] = False
        pool_indices = torch.where(mask)[0]
        a_pool_filtered = a_pool[mask]
        
        # Select most uncertain points using QbC
        if round == 0:
            selected = al.qbc_selection(ensemble, a_pool_filtered, n_select_first)
        else:
            selected = al.qbc_selection(ensemble, a_pool_filtered, n_select)

        # Delete current ensemble to free memory
        del ensemble
        gc.collect()
        torch.cuda.empty_cache()
            
        # Add to training set
        new_indices = pool_indices[selected]
        train_indices = torch.cat([train_indices, new_indices])
        a_train = a_pool[train_indices]
        u_train = u_pool[train_indices]
        
        # Retrain ensemble
        print(f"Retraining with {len(train_indices)} points...")
        # N = n_init + (round+1)*n_select
        if round == 0:
            N = n_init + n_select_first
        else:
            N = n_init + n_select_first + round*n_select
            
        if round+1 < n_rounds:
            ensemble = al.train_ensemble(a_train, u_train)
            # Compute Relative OOD Error - Save the first ensemble error
            error = al.relativeOODerror(ensemble[0])
            print(f'N: {N} | Relative OOD Error: {error.item()}')
            with open(f"ALLogs/QBC_trial{i}_log.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([N, error.item()])
        else:
            # Final model
            final_model, _, _ = al.train_model(a_train, u_train)
            # Compute Relative OOD Error
            error = al.relativeOODerror(final_model)
            print(error)
            with open(f"ALLogs/QBC_trial{i}_log.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([N, error.item()])
                    # Delete current ensemble to free memory

            # Delete final_model to free memory for next trial
            del final_model
            gc.collect()
            torch.cuda.empty_cache()