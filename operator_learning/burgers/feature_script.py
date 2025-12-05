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
    with open(f"ALLogs/feature_trial{i}_log.csv", "w", newline='') as f:
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
    
    # Train initial model (NOT ensemble - feature diversity uses single model)
    print("Training initial model...")
    model, _, _ = al.train_model(a_train, u_train)
    
    # Compute Relative OOD Error
    error = al.relativeOODerror(model)
    print(f'N: {n_init} | Relative OOD Error: {error.item()}')
    with open(f"ALLogs/feature_trial{i}_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([n_init, error.item()])
    
    # Track selected features across rounds (for already_selected_features parameter)
    all_selected_features = None
    
    for round in range(n_rounds):
        print(f"\nRound {round+1}/{n_rounds}")
        
        # Create pool (exclude already selected points)
        mask = torch.ones(a_pool.shape[0], dtype=torch.bool)
        mask[train_indices] = False
        pool_indices = torch.where(mask)[0]
        a_pool_filtered = a_pool[mask]
        
        # Select diverse points using feature diversity
        if round == 0:
            selected_in_pool, selected_features = al.feature_diversity_selection(
                model, 
                a_pool_filtered,  # Use filtered pool, not full pool
                n_select_first, 
                already_selected_features=all_selected_features
            )
        else: 
            selected_in_pool, selected_features = al.feature_diversity_selection(
                model, 
                a_pool_filtered,  # Use filtered pool, not full pool
                n_select, 
                already_selected_features=all_selected_features
            )
        # Update tracked features
        if all_selected_features is None:
            all_selected_features = selected_features.clone()
        else:
            all_selected_features = torch.vstack([all_selected_features, selected_features.to(all_selected_features.device)])
        
        # Add to training set
        new_indices = pool_indices[selected_in_pool]
        train_indices = torch.cat([train_indices, new_indices])
        a_train = a_pool[train_indices]
        u_train = u_pool[train_indices]
        
        # Retrain model
        print(f"Retraining with {len(train_indices)} points...")
        if round == 0:
            N = n_init + n_select_first
        else:
            N = n_init + n_select_first + round*n_select

        # Delete current model to free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train new model with all selected data
        model, _, _ = al.train_model(a_train, u_train)
        
        # Compute Relative OOD Error
        error = al.relativeOODerror(model)
        print(f'N: {N} | Relative OOD Error: {error.item()}')
        with open(f"ALLogs/feature_trial{i}_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([N, error.item()])