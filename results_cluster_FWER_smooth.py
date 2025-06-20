import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from scipy.ndimage import label
import os
from scipy.ndimage import gaussian_filter

# # # Load correlation maps
# r_audio = np.load("results/mc/correlation_map_mean_audio_opensmile_base_sar_iro_pro_sem_tom.npy")
# r_text = np.load("results/mc/correlation_map_mean_text_weighted_base_sar_iro_pro_sem_tom.npy")
# r_text_audio = np.load("results/mc/correlation_map_mean_audio_opensmile_text_weighted_base_sar_iro_pro_sem_tom.npy")

# # Load correlation maps
r_audio = np.load("results/normalized/correlation_map_mean_audio_opensmile_base_sar_iro_pro_sem_tom.npy")
r_text = np.load("results/normalized/correlation_map_mean_text_weighted_base_sar_iro_pro_sem_tom.npy")
r_text_audio = np.load("results/normalized/correlation_map_mean_audio_opensmile_text_weighted_base_sar_iro_pro_sem_tom.npy")


############################################################################################
# Load mask and affine
brain_mask = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\group_masks\group_mask\group_mask_threshold_0.25.nii.gz")
affine = brain_mask.affine
voxel_sizes = brain_mask.header.get_zooms()[:3]
brain_mask = brain_mask.get_fdata() > 0
# Define smoothing
fwhm = 6.0
sigma_mm = fwhm / np.sqrt(8 * np.log(2))
sigma_vox = [sigma_mm / vs for vs in voxel_sizes]
def smooth_within_mask(data, mask, sigma_vox):
    data_masked = np.zeros_like(data)
    data_masked[mask] = data[mask]
    smoothed = gaussian_filter(data_masked, sigma=sigma_vox)
    smoothed[~mask] = 0
    return smoothed
# Apply mask-aware smoothing
r_audio = smooth_within_mask(r_audio, brain_mask, sigma_vox)
r_text = smooth_within_mask(r_text, brain_mask, sigma_vox)
r_text_audio = smooth_within_mask(r_text_audio, brain_mask, sigma_vox)


# Check zeros within brain
print(f"Zero voxels in r_audio (within brain): {np.sum(r_audio[brain_mask] == 0) / np.sum(brain_mask):.2%}")
print(f"Zero voxels in r_text (within brain): {np.sum(r_text[brain_mask] == 0) / np.sum(brain_mask):.2%}")
print(f"Zero voxels in r_text_audio (within brain): {np.sum(r_text_audio[brain_mask] == 0) / np.sum(brain_mask):.2%}")

# Step 1: Get the original shape
X, Y, Z = r_audio.shape

# Step 2: Compute observed improvement
delta_r = r_text_audio - np.maximum(r_audio, r_text)

valid_mask = (r_text != 0) & (r_audio != 0) & (r_text_audio != 0)
delta_r = np.zeros_like(r_text)
delta_r[valid_mask] = r_text[valid_mask] - np.maximum(r_text_audio[valid_mask], r_audio[valid_mask])


# Step 3: Flatten only brain voxels
brain_mask_flat = brain_mask.ravel()
valid_voxels = brain_mask_flat
V = np.sum(valid_voxels)  # Number of brain voxels
r_audio_flat = r_audio.reshape(-1)[valid_voxels]
r_text_flat = r_text.reshape(-1)[valid_voxels]
r_text_audio_flat = r_text_audio.reshape(-1)[valid_voxels]
delta_r_obs_flat = delta_r.reshape(-1)[valid_voxels]

# Step 4: Permutation test
num_permutations = 5000
delta_r_perm = np.zeros((V, num_permutations))
for i in range(num_permutations):
    permuted_indices = np.random.permutation(V)
    r_audio_perm = r_audio_flat[permuted_indices]
    r_text_perm = r_text_flat[permuted_indices]
    r_text_audio_perm = r_text_audio_flat[permuted_indices]
    delta_r_perm[:, i] = r_text_audio_perm - np.maximum(r_audio_perm, r_text_perm)

# Step 5: Compute uncorrected p-values
p_values = np.mean(delta_r_perm >= delta_r_obs_flat[:, np.newaxis], axis=1)

# Step 6: Apply cluster-forming threshold
p_values_3d = np.zeros((X, Y, Z))
p_values_3d[brain_mask] = p_values
initial_mask_3d = p_values_3d < 0.01
p_values_nifti = nib.Nifti1Image(p_values_3d, affine)

# Step 7: Identify clusters in observed data
labeled_array, num_clusters = label(initial_mask_3d, structure=np.ones((3, 3, 3)))
observed_cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)

# Step 8: Compute cluster sizes in permutations
max_cluster_sizes_perm = np.zeros(num_permutations)
for i in range(num_permutations):
    perm_mask_flat = delta_r_perm[:, i] >= delta_r_obs_flat
    perm_mask_3d = np.zeros((X, Y, Z), dtype=bool)
    perm_mask_3d[brain_mask] = perm_mask_flat
    labeled_perm, _ = label(perm_mask_3d, structure=np.ones((3, 3, 3)))
    cluster_sizes_perm = np.bincount(labeled_perm.ravel())[1:]
    max_cluster_sizes_perm[i] = np.max(cluster_sizes_perm) if cluster_sizes_perm.size > 0 else 0

# Step 9: Compute cluster-level p-values
cluster_p_values = np.array([np.mean(max_cluster_sizes_perm >= size) for size in observed_cluster_sizes])
significant_clusters = np.where(cluster_p_values < 0.05)[0] + 1  # Cluster labels start at 1
clustered_mask_3d = np.isin(labeled_array, significant_clusters)

# Step 10: Create significance map (all significant clusters)
significance_map_3d = np.zeros_like(delta_r)
significance_map_3d[clustered_mask_3d] = delta_r[clustered_mask_3d]

# Step 11 (Optional): Filter for positive improvements
positive_mask = delta_r > 0
significance_map_positive_3d = np.zeros_like(delta_r)
significance_map_positive_3d[clustered_mask_3d & positive_mask] = delta_r[clustered_mask_3d & positive_mask]

# Print cluster summary
if len(significant_clusters) == 0:
    print("No significant clusters found (p < 0.05).")
else:
    print(f"Number of significant clusters (all): {len(significant_clusters)}")
    print(f"Cluster p-values: {cluster_p_values}")
    # Count clusters with positive delta
    positive_clusters = np.unique(labeled_array[clustered_mask_3d & positive_mask])
    positive_clusters = positive_clusters[positive_clusters > 0]  # Exclude background
    print(f"Number of significant clusters with positive delta: {len(positive_clusters)}")

# Save NIfTI for all significant clusters
significance_nifti = nib.Nifti1Image(significance_map_3d, affine)

# Save NIfTI for positive clusters only
significance_positive_nifti = nib.Nifti1Image(significance_map_positive_3d, affine)
nib.save(significance_positive_nifti, 'results/final_results/results_sar_iro_pro_sem_tom_mc_opensmile_smooth.nii')

# Plot glass brain (all significant clusters)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    significance_positive_nifti,
    threshold=0,
    title='Significant Delta R² (Text+Audio vs Max(Text,Audio))\nFWER-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()
plt.savefig('results/final_results/plot_sar_iro_pro_sem_tom_mc_opensmile_smooth.png', dpi=300)

