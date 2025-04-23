import torch
import numpy as np
from scipy.special import gamma
import os

def random_input_points(num_centers, dimension):
    # Generate random points uniformly in the range [-1, 1] in D dimensions
    return np.random.uniform(low=-1.0, high=1.0, size=(num_centers, dimension))

def generate_gaussian_parameters(num_centers, amplitude=None, amplitude_range=(0.5, 1.5), sigma_range=(0.5, 1.5)):
    if amplitude is not None:
        # Use the fixed amplitude for all Gaussians
        amplitudes = np.full(num_centers, amplitude)
    else:
        # Generate random amplitudes within the specified range
        amplitudes = np.random.uniform(low=amplitude_range[0], high=amplitude_range[1], size=num_centers)
    # Generate random sigmas
    sigmas = np.random.uniform(low=sigma_range[0], high=sigma_range[1], size=num_centers)
    return amplitudes, sigmas

# Function to generate and save datasets
def generate_and_save_datasets(dimensions, num_centers, output_folder, num_sets, base_seed, amplitude=None, amplitude_range=(0.5, 1.5), sigma_range=(0.5, 1.5)):
    for set_num in range(1, num_sets + 1):
        # Create output subfolder
        set_folder = os.path.join(output_folder, f'dataset_{set_num}')
        os.makedirs(set_folder, exist_ok=True)
        
        for dimension, num_centers_dim in zip(dimensions, num_centers):
            # Set a unique seed for this dataset
            seed = base_seed + set_num + dimension*20  # Ensures each dataset has a unique seed
            np.random.seed(seed)
            
            # Generate random points
            points = random_input_points(num_centers_dim, dimension)
            amplitudes, sigmas = generate_gaussian_parameters(num_centers_dim, amplitude, amplitude_range, sigma_range=(0.5, 1.5))
    
            # Combine points, amplitudes, and sigmas into a single dataset
            dataset = np.column_stack((points, amplitudes, sigmas))
    
            # Save the dataset to a text file
            filename = os.path.join(set_folder, f'dataset_dim_{dimension}.txt')
            np.savetxt(filename, dataset, fmt='%.6f', header=' '.join([f'x{i}' for i in range(dimension)] + ['amplitude', 'sigma']))
    
            print(f"Dataset for dimension {dimension} with {num_centers_dim} centers saved to '{filename}'")

## Auxiliary functions to estimate the growth of points in high dimensions
def sphere_volume(r, d):
    return (np.pi ** (d / 2)) / gamma(d / 2 + 1) * (r ** d)

def average_sphere_volume(r_min, r_max, d):
    if r_min == r_max:
        # If bounds are equal, return the volume for that radius
        return (np.pi ** (d / 2)) / gamma(d / 2 + 1) * (r_min ** d)
    else:
        # Compute the average volume over the uniform distribution of r
        prefactor = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
        integral = (r_max ** (d + 1) - r_min ** (d + 1)) / (d + 1)
        return prefactor * integral / (r_max - r_min)

def compute_volume_ratios(dimensions, amplitude):
    """
    Compute the ratio 2^n / volume for each dimension, where volume is the volume of a sphere with radius equal to amplitude.

    Args:
        dimensions (list): List of dimensions.
        amplitude (float): Radius of the sphere.

    Returns:
        np.array: Array of ratios 2^n / volume for each dimension.
    """
    ratios = []
    for n in dimensions:
        volume = sphere_volume(amplitude, n)
        ratio = (2 ** n) / volume
        ratios.append(ratio)
    return np.array(ratios)

def random_input_points(num_samples, dimension, seed=None):
    if seed is not None:
        # Save current random state
        rng_state = torch.get_rng_state()
        # Set new seed
        torch.manual_seed(seed)
    xy = torch.rand((num_samples, dimension)) * 2 - 1  # Scale to [-1, 1] in N dimensions
    if seed is not None:
        # Restore original random state
        torch.set_rng_state(rng_state)
    return xy

def generate_random_transform_matrix(dimension, sparsity=0.5, scaling_range=(-1.5, 1.5), seed=None):
    """
    Generate a random transformation matrix with mostly zeros and some swaps/scales.
    
    Args:
        dimension (int): Size of the matrix (dimension x dimension).
        sparsity (float): Probability that a given element is zero (default: 0.5).
        scaling_range (tuple): Range for random scaling values (default: (-1.5, 1.5)).
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.
    
    Returns:
        torch.Tensor: A transformation matrix of shape (dimension, dimension).
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    transform_matrix = torch.zeros((dimension, dimension))
    
    # Phase 1: Random initialization with sparsity
    for i in range(dimension):
        for j in range(dimension):
            if torch.rand(1).item() > sparsity:
                scale = torch.empty(1).uniform_(*scaling_range).item()
                transform_matrix[i, j] = scale
    
    # Phase 2: Ensure at least one non-zero per row
    for i in range(dimension):
        if torch.all(transform_matrix[i] == 0):
            j = torch.randint(0, dimension, (1,)).item()
            scale = torch.empty(1).uniform_(*scaling_range).item()
            transform_matrix[i, j] = scale
    
    # Phase 3: Ensure at least one non-zero per column
    for j in range(dimension):
        if torch.all(transform_matrix[:, j] == 0):
            i = torch.randint(0, dimension, (1,)).item()
            scale = torch.empty(1).uniform_(*scaling_range).item()
            transform_matrix[i, j] = scale
    
    return transform_matrix

def vector_function(input_data, centers, amplitudes, sigmas, transform_matrix=None, center_batch_size=200):
    """
    Compute the vector field at the given input points using Gaussian contributions, with optional linear transformation.
    
    Args:
        input_data (torch.Tensor): Tensor of shape (num_samples, dimension) containing the input points.
        centers (torch.Tensor): Tensor of shape (num_centers, dimension) containing the Gaussian centers.
        amplitudes (torch.Tensor): Tensor of shape (num_centers,) containing the Gaussian amplitudes.
        sigmas (torch.Tensor): Tensor of shape (num_centers,) containing the Gaussian sigmas.
        transform_matrix (torch.Tensor): Tensor of shape (dimension, dimension) describing how to mix the output components. 
                                        If None, no transformation is applied.
        center_batch_size (int): Maximum number of centers to process at a time. Default is 200.
    
    Returns:
        force (torch.Tensor): Tensor of shape (num_samples, dimension) containing the transformed vector field values.
    """
    num_samples = input_data.shape[0]
    dimension = input_data.shape[1]
    num_centers = centers.shape[0]

    # Initialize the output tensor to store the force values
    force = torch.zeros((num_samples, dimension), device=input_data.device)

    # Process centers, amplitudes, and sigmas in batches
    for i in range(0, num_centers, center_batch_size):
        # Get the current batch of centers, amplitudes, and sigmas
        batch_end = min(i + center_batch_size, num_centers)
        batch_centers = centers[i:batch_end]  # Shape: (batch_size, dimension)
        batch_amplitudes = amplitudes[i:batch_end]  # Shape: (batch_size,)
        batch_sigmas = sigmas[i:batch_end]  # Shape: (batch_size,)

        # Expand input_data and batch_centers for broadcasting
        input_data_expanded = input_data.unsqueeze(1)  # Shape: (num_samples, 1, dimension)
        centers_expanded = batch_centers.unsqueeze(0)  # Shape: (1, batch_size, dimension)

        # Compute differences between input points and batch centers
        diff = input_data_expanded - centers_expanded  # Shape: (num_samples, batch_size, dimension)

        # Compute the squared distances and Gaussian exponent terms
        squared_distances = torch.sum(diff**2, dim=2)  # Shape: (num_samples, batch_size)
        exp_terms = torch.exp(-squared_distances / (2 * batch_sigmas**2))  # Shape: (num_samples, batch_size)

        # Compute the gradients for each dimension
        grad = (batch_amplitudes / batch_sigmas**2).unsqueeze(0).unsqueeze(-1) * diff * exp_terms.unsqueeze(-1)  # Shape: (num_samples, batch_size, dimension)

        # Sum contributions from the current batch of centers
        batch_force = -torch.sum(grad, dim=1)  # Shape: (num_samples, dimension)

        # Accumulate the batch results in the output tensor
        force += batch_force

    # Apply transformation if specified
    if transform_matrix is not None:
        force = torch.einsum('ij,sj->si', transform_matrix, force)  # Shape: (num_samples, dimension)

    return force