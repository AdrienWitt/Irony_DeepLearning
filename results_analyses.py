import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
import os
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from nilearn.image import smooth_img


# Load correlation maps (Shape: X × Y × Z, already mean across folds)
r_audio = np.load("results/correlation_map_mean_audio_base.npy")  # Shape (X, Y, Z)
r_text = np.load("results/correlation_map_mean_text_base.npy")    # Shape (X, Y, Z)
r_text_audio = np.load("results/correlation_map_mean_text_audio_base.npy")  # Shape (X, Y, Z)

data_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted"  # Replace with the path to your brain data

all_affines = []
for subject in os.listdir(data_folder):
    subject_path = os.path.join(data_folder, subject)
    for file in os.listdir(subject_path):  # Fixed variable name 'files' to 'file'
        if file.endswith('.nii') or file.endswith('.nii.gz'):  # Check for NIfTI files
            file_path = os.path.join(subject_path, file)  # Fixed os.path.join syntax
            nifti_img = nib.load(file_path)
            all_affines.append(nifti_img.affine)

affine = np.array(np.mean(all_affines, axis=0))
# Proceed with your analysis (Step 2 onward)
delta_r_obs_3d = r_text_audio - np.maximum(r_audio, r_text)

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

# Step 6: Compute p-values for all voxels
p_values = np.mean(delta_r_perm >= delta_r_obs_flat[:, np.newaxis], axis=1)  # Shape (V,)

# Step 7: Adjust p-values for ALL comparisons using FDR (Benjamini-Hochberg)
reject, p_values_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# Step 8: Find significant voxels (adjusted p < 0.05) among positive delta_r
significant_mask_flat = (p_values_adjusted < 0.05) & positive_mask_flat

# Check the difference between original and adjusted p-values
print("Original p-values (first 10 voxels):")
print(p_values[:10])
print("\nAdjusted p-values (first 10 voxels):")
print(p_values_adjusted[:10])

# Calculate summary statistics
print(f"\nMean original p-value: {np.mean(p_values):.6f}")
print(f"Mean adjusted p-value: {np.mean(p_values_adjusted):.6f}")

# How many significant voxels before and after correction?
sig_before = np.sum(p_values < 0.05)
sig_after = np.sum(p_values_adjusted < 0.05)
print(f"\nSignificant voxels before correction: {sig_before} ({sig_before/len(p_values)*100:.2f}%)")
print(f"Significant voxels after correction: {sig_after} ({sig_after/len(p_values)*100:.2f}%)")

# Reshape the significant mask back to 3D space
significant_mask_3d = significant_mask_flat.reshape(X, Y, Z)
# Create a significance map showing the delta_r values only where significant
significance_map_3d = np.zeros_like(delta_r_obs_3d)
significance_map_3d[significant_mask_3d] = delta_r_obs_3d[significant_mask_3d]

# Create a new NIfTI image with the significance map and template's affine
significance_nifti = nib.Nifti1Image(significance_map_3d,affine)

# Save the significance map as a NIfTI file
nib.save(significance_nifti, 'results/significant_improvements_fdr_corrected.nii')

# Plot using nilearn
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(significance_nifti, threshold=0, 
                         title='Significant improvement (text+audio > max(text,audio))\nFDR-corrected p < 0.05',
                         colorbar=True, plot_abs=False,
                         display_mode='ortho', axes=ax)
plt.tight_layout()
plt.savefig('results/significant_improvements_glass_brain.png', dpi=300)
plt.close()


