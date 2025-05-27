import torch

def gaussian_kernel(x, y, sigma=5e-2):
    """
    Computes the Gaussian (RBF) kernel between two sets of vectors.
    
    Args:
        x (torch.Tensor): A tensor of shape (n_samples_x, d).
        y (torch.Tensor): A tensor of shape (n_samples_y, d).
        sigma (float): The standard deviation for the Gaussian kernel.
    
    Returns:
        torch.Tensor: A tensor of shape (n_samples_x, n_samples_y) containing pairwise kernel values.
    """
    # Squared Euclidean distance between points
    xx = torch.sum(x ** 2, dim=1).view(-1, 1)
    yy = torch.sum(y ** 2, dim=1).view(1, -1)
    dist = xx + yy - 2 * torch.mm(x, y.t())
    
    # Gaussian kernel (RBF kernel)
    return torch.exp(-torch.relu(dist) / (2 * sigma ** 2))

def laplace_kernel(x, y, sigma=5e-2):
    """
    Computes the Laplace kernel between two sets of vectors.
    
    Args:
        x (torch.Tensor): A tensor of shape (n_samples_x, d).
        y (torch.Tensor): A tensor of shape (n_samples_y, d).
        sigma (float): The lengthscale for the kernel.
    
    Returns:
        torch.Tensor: A tensor of shape (n_samples_x, n_samples_y) containing pairwise kernel values.
    """
    # pairwise l^1 distance between points
    dist = torch.cdist(x,y,p=1.0)
    
    return torch.exp(-dist / sigma)

def energy_kernel(x, y):
    """
    Computes the negative Euclidean distance kernel between two sets of vectors.
    
    Args:
        x (torch.Tensor): A tensor of shape (n_samples_x, d).
        y (torch.Tensor): A tensor of shape (n_samples_y, d).
    
    Returns:
        torch.Tensor: A tensor of shape (n_samples_x, n_samples_y) containing pairwise kernel values.
    """
    # Squared Euclidean distance between points
    xx = torch.sum(x ** 2, dim=1).view(-1, 1)
    yy = torch.sum(y ** 2, dim=1).view(1, -1)
    dist = xx + yy - 2 * torch.mm(x, y.t())
    
    return -torch.sqrt(torch.relu(dist))