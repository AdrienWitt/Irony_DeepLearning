import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
import os
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
# Load correlation maps (Shape: X × Y × Z, already mean across folds)
r_audio = np.load("results/correlation_map_mean_audio_base.npy")  # Shape (X, Y, Z)
r_text = np.load("results/correlation_map_mean_text_base.npy")    # Shape (X, Y, Z)
r_text_audio = np.load("results/correlation_map_mean_text_audio_base.npy")  # Shape (X, Y, Z)

# Step 1: Get the original shape
X, Y, Z = r_audio.shape  # Brain volume dimensions (no folds)

# Step 2: Compute observed improvement per voxel in 3D
delta_r_obs_3d = r_text_audio - np.maximum(r_audio, r_text)  # Shape (X, Y, Z)

# Step 3: Filter for positive delta_r before flattening
positive_mask = delta_r_obs_3d > 0  # Boolean mask for positive improvements
delta_r_positive_3d = delta_r_obs_3d * positive_mask  # Keep only positive values

# Step 4: Flatten spatial dimensions
V = X * Y * Z  # Total number of voxels
r_audio_flat = r_audio.reshape(V)
r_text_flat = r_text.reshape(V)
r_text_audio_flat = r_text_audio.reshape(V)
delta_r_obs_flat = delta_r_obs_3d.reshape(V)
positive_mask_flat = positive_mask.reshape(V)

# Step 5: Permutation test
num_permutations = 1000
delta_r_perm = np.zeros((V, num_permutations))

for i in range(num_permutations):
    permuted_indices = np.random.permutation(V)
    r_audio_perm = r_audio_flat[permuted_indices]
    r_text_perm = r_text_flat[permuted_indices]
    r_text_audio_perm = r_text_audio_flat[permuted_indices]
    delta_r_perm[:, i] = r_text_audio_perm - np.maximum(r_audio_perm, r_text_perm)

# Step 6: Compute p-values
p_values = np.mean(delta_r_perm >= delta_r_obs_flat[:, np.newaxis], axis=1)  # Shape (V,)

# Step 7: Adjust p-values for multiple comparisons using FDR (Benjamini-Hochberg)
p_values_adjusted = np.ones_like(p_values)  # Initialize with 1s (non-significant)
p_values_positive = p_values[positive_mask_flat]  # Extract p-values for positive delta_r
reject, p_values_adj_positive, _, _ = multipletests(p_values_positive, alpha=0.05, method='fdr_bh')
p_values_adjusted[positive_mask_flat] = p_values_adj_positive  # Assign adjusted p-values

# Step 8: Find significant voxels (adjusted p < 0.05) among positive delta_r
significant_mask_flat = (p_values_adjusted < 0.05) & positive_mask_flat
significant_voxel_indices = np.where(significant_mask_flat)[0]

# Step 9: Get top 25% of positive delta_r values
positive_delta_r_values = delta_r_obs_flat[positive_mask_flat]
threshold_75 = np.percentile(positive_delta_r_values, 75)
top_25_mask_flat = (delta_r_obs_flat > threshold_75) & positive_mask_flat
top_25_indices = np.where(top_25_mask_flat)[0]

# Step 10: Create a list of significant voxels with coordinates, p-values, and delta_r
coords = np.unravel_index(significant_voxel_indices, (X, Y, Z))
significant_data = {
    'x': coords[0],
    'y': coords[1],
    'z': coords[2],
    'adjusted_p_value': p_values_adjusted[significant_mask_flat],
    'delta_r': delta_r_obs_flat[significant_mask_flat]
}
significant_df = pd.DataFrame(significant_data)

