import torch
import math
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["CUDA_PATH"] = "/usr/local/cuda-11.8"
os.environ["PATH"] = "/usr/local/cuda-11.8/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
import pykeops
# pykeops.clean_pykeops()

class ActiveLearning:
    def __init__(self,a_valid,u_valid,device,n=8192,L=2*math.pi):
        self.a_valid = a_valid
        self.u_valid = u_valid
        self.device = device
        self.dx = L/n
        self.L = L
        self.n = n # Number of discretization points 
        self.trunk = torch.linspace(0, L, n+1, device=device)[:-1].unsqueeze(-1) 
    
    def train_model(self, a, u):
        N_samples = a.shape[0]
        
        # Train/test split - use 90/10 since we have limited data
        indices = torch.randperm(N_samples)
        ntrain = int(0.9 * N_samples) 
        a_train, a_test = a[indices[:ntrain]], a[indices[ntrain:]]
        u_train, u_test = u[indices[:ntrain]], u[indices[ntrain:]]
            
        # DeepXDE data object
        data = dde.data.TripleCartesianProd(
            X_train=(a_train, self.trunk),
            y_train=u_train,
            X_test=(a_test, self.trunk),
            y_test=u_test,
        )
        
        # Smaller architecture to reduce overfitting
        p = 128  
        net = dde.nn.pytorch.DeepONetCartesianProd(
            layer_sizes_branch=[self.n, 1024, 512, 256, 128, p],
            layer_sizes_trunk=[1, 128, 128, 128, 128, p],
            activation="relu",
            kernel_initializer="Glorot normal"
        )
        net.to(self.device)
        
        # Compile model with learning rate decay
        model = dde.Model(data, net)
        model.compile(
            "adam",
            lr=1e-3,
            loss="mean l2 relative error",
            decay=("inverse time", 2000, 0.9),
        )
        
        # Add L2 regularization via weight decay
        for param_group in model.opt.param_groups:
            param_group['weight_decay'] = 1e-3
        
        print('STARTING MODEL TRAINING')
        
        # Train with early stopping callback - monitor loss_test
        checker = dde.callbacks.EarlyStopping(
            min_delta=1e-4,
            patience=1000,
            baseline=None,
            monitor='loss_test' 
        )
        
        losshistory, train_state = model.train(
            iterations=20000,
            display_every=1000,
            callbacks=[checker],
        )
        
        print(f"\nFinal train loss: {train_state.best_loss_train:.2e}")
        print(f"Final test loss: {train_state.best_loss_test:.2e}")
        
        return model, losshistory, train_state

    def train_ensemble(self, a_train, u_train, n_models=4):
        """Train multiple models with different initializations"""
        ensemble = []
        for i in range(n_models):
            # Each model gets different random initialization
            model,_,_ = self.train_model(a_train,u_train)
            ensemble.append(model)
        
        return ensemble

    def qbc_selection(self, ensemble, a_pool, n_select):
        """
        Query-by-Committee selection using ensemble variance
        Returns indices of n_select points with highest uncertainty
        """
        n_pool = a_pool.shape[0]
        n_models = len(ensemble)
        
        # Get predictions from all ensemble members
        all_preds = []
        with torch.no_grad():
            for model in ensemble:
                preds = model.net((a_pool, self.trunk))  # (n_pool, self.n)
                all_preds.append(preds)
        
        # Stack: (n_models, n_pool, self.n)
        all_preds = torch.stack(all_preds)
        
        # Compute mean prediction across ensemble
        mean_pred = all_preds.mean(dim=0)  # (n_pool, self.n)
        
        # Compute variance (uncertainty) for each point in pool
        # average squared deviation from mean
        uncertainties = torch.zeros(n_pool, device=self.device)
        
        for i in range(n_pool):
            # For each candidate function a_i
            variance_sum = 0
            for m in range(n_models):
                # ||predicted_u_{i,m} - mean_predicted_u_i||^2_2 at each spatial point
                diff = all_preds[m, i, :] - mean_pred[i, :]
                variance_sum += self.norm(diff)**2  # Using L2 norm with dx
            
            uncertainties[i] = variance_sum / n_models
        
        # Select points with highest uncertainty
        _, indices = torch.topk(uncertainties, n_select)
        
        return indices

    def gaussian_sketch(self, features, target_dim=32):
        """Project features (n, p) -> (n, target_dim) with Gaussian matrix U / sqrt(p')."""
        n, p = features.shape
        gen = torch.Generator(device=features.device)
        gen.manual_seed(42)
        U = torch.randn((target_dim, p), generator=gen, device=features.device)
        U = U / math.sqrt(target_dim)   # divide by sqrt(p') per paragraph
        return features @ U.t()         # (n, target_dim)
    
    
    def extract_deeponet_features(self, model, a_pool, target_dim=32):
        """
        Simple feature extractor for DeepONet.
        """
        net = model.net
        branch = net.branch
        trunk = net.trunk

        with torch.no_grad():
            # Branch features: handle (batch, T, ...) or (batch, ...)
            f = branch(a_pool)
            branch_feats = f.reshape(f.shape[0], -1)  # (batch, p)
    
            trunk_out = trunk(self.trunk)             # expect (n_spatial, p_trunk)
            trunk_out = trunk_out.reshape(trunk_out.shape[0], -1)
            trunk_avg = trunk_out.mean(0, keepdim=True)    # (1, p_trunk)
            trunk_expanded = trunk_avg.expand(branch_feats.shape[0], -1)  # (batch, p_trunk)
    
            combined = torch.cat([branch_feats, trunk_expanded], dim=1)  # (batch, p_total)
            sketched = self.gaussian_sketch(combined, target_dim=target_dim)
    
        return sketched  # (batch, target_dim)
    
    def feature_diversity_selection(self, model, a_pool, n_select, already_selected_features = None, sketch_dim=32):
        """
        Greedy k-center (CoreSet) on Gaussian-sketched features.
        """
        pool_features = self.extract_deeponet_features(model, a_pool, target_dim=sketch_dim)
        pool_features = pool_features / (pool_features.norm(dim=1, keepdim=True) + 1e-8)
    
        N = pool_features.shape[0]
        device = pool_features.device
    
        # initialize selected set
        if already_selected_features is None or already_selected_features.numel() == 0:
            start = torch.randint(0, N, (1,), device=device).item()
            selected_indices = [start]
            selected_feats = [pool_features[start]]
        else:
            # already_selected_features should be a tensor (M, sketch_dim)
            selected_indices = []
            selected_feats = [f for f in already_selected_features]  # torch iteration yields rows
    
        # compute min distances to current set
        min_dists = torch.full((N,), float("inf"), device=device)
        for s in selected_feats:
            d = torch.norm(pool_features - s.unsqueeze(0), dim=1)
            min_dists = torch.minimum(min_dists, d)
    
        while len(selected_indices) < n_select:
            # mask already chosen
            for idx in selected_indices:
                min_dists[idx] = -1.0
            new_idx = int(torch.argmax(min_dists).item())
            selected_indices.append(new_idx)
            new_feat = pool_features[new_idx]
            selected_feats.append(new_feat)
            # update min_dists
            d = torch.norm(pool_features - new_feat.unsqueeze(0), dim=1)
            min_dists = torch.minimum(min_dists, d)
    
        selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
        selected_features_tensor = torch.stack(selected_feats[:n_select], dim=0)  # (n_select, sketch_dim)
    
        return selected_indices_tensor, selected_features_tensor
        
    def norm(self, a):
        '''
        This function computes the L2 norm of each of the N periodic functions.
        Each function is given as a discretized vector with grid spacing dx.
        a is size [N,n]
        The output is a vector of size N.
        '''
        return torch.sqrt(torch.sum(a**2, dim=-1) * self.dx)

    def relativeOODerror(self,model):
        with torch.no_grad():
            num = torch.mean(self.norm(self.u_valid - model.net((self.a_valid,self.trunk)))**2)
        den = torch.mean(self.norm(self.u_valid)**2)
        return torch.sqrt(num/den)