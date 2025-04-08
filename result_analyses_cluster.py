# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:45:04 2025

@author: adywi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
import os
import nibabel as nib
from nilearn import plotting, datasets, image
from scipy.ndimage import label

# Load correlation maps
r_audio = np.load("results/correlation_map_mean_audio_base.npy")
r_text = np.load("results/correlation_map_mean_text_base.npy")
r_text_audio = np.load("results/correlation_map_mean_text_audio_base.npy")

data_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted"  # Replace with the path to your brain data
##compute average affine
all_affines = []
for subject in os.listdir(data_folder):
    subject_path = os.path.join(data_folder, subject)
    for file in os.listdir(subject_path):  # Fixed variable name 'files' to 'file'
        if file.endswith('.nii') or file.endswith('.nii.gz'):  # Check for NIfTI files
            file_path = os.path.join(subject_path, file)  # Fixed os.path.join syntax
            nifti_img = nib.load(file_path)
            all_affines.append(nifti_img.affine)

affine = np.array(np.mean(all_affines, axis=0))

# Smooth the correlation maps
r_audio_nifti = nib.Nifti1Image(r_audio, affine)
r_text_nifti = nib.Nifti1Image(r_text, affine)
r_text_audio_nifti = nib.Nifti1Image(r_text_audio, affine)
fwhm = 6.0
r_audio = image.smooth_img(r_audio_nifti, fwhm=fwhm).get_fdata()
r_text = image.smooth_img(r_text_nifti, fwhm=fwhm).get_fdata()
r_text_audio = image.smooth_img(r_text_audio_nifti, fwhm=fwhm).get_fdata()

# Step 1: Get the original shape
X, Y, Z = r_audio.shape

# Step 2: Compute observed improvement
delta_r_obs_3d = r_text_audio - np.maximum(r_audio, r_text)

# Step 3: Filter for positive delta_r
positive_mask = delta_r_obs_3d > 0
delta_r_positive_3d = delta_r_obs_3d * positive_mask

# Step 4: Flatten
V = X * Y * Z
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

# Step 6: Compute uncorrected p-values
p_values = np.mean(delta_r_perm >= delta_r_obs_flat[:, np.newaxis], axis=1)

# Step 7: Apply cluster-forming threshold
cluster_threshold = 0.05
initial_mask_3d = (p_values.reshape(X, Y, Z) < cluster_threshold) & positive_mask

# Step 8: Identify clusters in observed data
labeled_array, num_clusters = label(initial_mask_3d)
observed_cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)

# Step 9: Compute cluster sizes in permutations
max_cluster_sizes_perm = np.zeros(num_permutations)
for i in range(num_permutations):
    perm_mask_3d = (delta_r_perm[:, i] >= delta_r_obs_flat) & positive_mask_flat
    perm_mask_3d = perm_mask_3d.reshape(X, Y, Z)
    labeled_perm, _ = label(perm_mask_3d)
    cluster_sizes_perm = np.bincount(labeled_perm.ravel())[1:]
    max_cluster_sizes_perm[i] = np.max(cluster_sizes_perm) if cluster_sizes_perm.size > 0 else 0

# Step 10: Compute cluster-level p-values
cluster_p_values = np.array([np.mean(max_cluster_sizes_perm >= size) for size in observed_cluster_sizes])
significant_clusters = np.where(cluster_p_values < 0.05)[0] + 1  # Cluster labels start at 1
clustered_mask_3d = np.isin(labeled_array, significant_clusters)

# Step 11: Update significance map
significance_map_3d = np.zeros_like(delta_r_obs_3d)
significance_map_3d[clustered_mask_3d] = delta_r_obs_3d[clustered_mask_3d]


# Create and save NIfTI
significance_nifti = nib.Nifti1Image(significance_map_3d, affine)
nib.save(significance_nifti, 'results/significant_improvements_cluster_corrected.nii')

# Visualize
min_delta = np.min(significance_map_3d[significance_map_3d > 0]) if np.any(significance_map_3d > 0) else 0
plotting.plot_glass_brain(
    significance_nifti,
    threshold=min_delta,
    title='Significant Clusters (Text+Audio > Max(Text, Audio))\nCluster-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r'
)
plt.savefig('results/significant_improvements_glass_brain_cluster.png', dpi=300)
plt.close()

# Print summary
print(f"Number of significant clusters: {len(significant_clusters)}")
print(f"Cluster p-values: {cluster_p_values}")