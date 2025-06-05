import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from scipy.ndimage import label
import os
from statsmodels.stats.multitest import fdrcorrection

# Load correlation maps
r_audio = np.load("results/mc/correlation_map_mean_audio_opensmile_base_sar_iro_pro_sem_tom.npy")
r_text = np.load("results/mc/correlation_map_mean_text_weighted_base_sar_iro_pro_sem_tom.npy")
r_text_audio = np.load("results/mc/correlation_map_mean_audio_opensmile_text_weighted_base_sar_iro_pro_sem_tom.npy")
brain_mask = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\group_masks\group_mask\group_mask_threshold_0.25.nii.gz")


affine = brain_mask.affine

# Create NIfTI images
r_audio_nifti = nib.Nifti1Image(r_audio, affine)
r_text_nifti = nib.Nifti1Image(r_text, affine)
r_text_audio_nifti = nib.Nifti1Image(r_text_audio, affine)
brain_mask = brain_mask.get_fdata() > 0


# Smooth and apply brain mask
fwhm = 6.0
r_audio = image.smooth_img(r_audio_nifti, fwhm=fwhm).get_fdata() * brain_mask
r_text = image.smooth_img(r_text_nifti, fwhm=fwhm).get_fdata() * brain_mask
r_text_audio = image.smooth_img(r_text_audio_nifti, fwhm=fwhm).get_fdata() * brain_mask

# Check zeros within brain
print(f"Zero voxels in r_audio (within brain): {np.sum(r_audio[brain_mask] == 0) / np.sum(brain_mask):.2%}")
print(f"Zero voxels in r_text (within brain): {np.sum(r_text[brain_mask] == 0) / np.sum(brain_mask):.2%}")
print(f"Zero voxels in r_text_audio (within brain): {np.sum(r_text_audio[brain_mask] == 0) / np.sum(brain_mask):.2%}")

# Step 1: Get the original shape
X, Y, Z = r_audio.shape

# Step 2: Compute observed improvement
delta_r_obs_3d = r_text_audio - np.maximum(r_audio, r_text)

# Step 3: Flatten only brain voxels
brain_mask_flat = brain_mask.ravel()
valid_voxels = brain_mask_flat
V = np.sum(valid_voxels)  # Number of brain voxels
r_audio_flat = r_audio.reshape(-1)[valid_voxels]
r_text_flat = r_text.reshape(-1)[valid_voxels]
r_text_audio_flat = r_text_audio.reshape(-1)[valid_voxels]
delta_r_obs_flat = delta_r_obs_3d.reshape(-1)[valid_voxels]

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

# Step 6: Apply FDR correction
_, p_values_fdr = fdrcorrection(p_values, alpha=0.05)

# Step 7: Create significance map for FDR-corrected voxels (all)
p_values_fdr_3d = np.zeros((X, Y, Z))
p_values_fdr_3d[brain_mask] = p_values_fdr
significant_fdr_mask_3d = (p_values_fdr_3d < 0.05) & brain_mask  # Explicitly mask with brain_mask
significance_map_fdr_3d = np.zeros_like(delta_r_obs_3d)
significance_map_fdr_3d[significant_fdr_mask_3d] = delta_r_obs_3d[significant_fdr_mask_3d]

# Step 8: Create significance map for FDR-corrected voxels with positive delta
positive_mask = delta_r_obs_3d > 0
significance_map_fdr_positive_3d = np.zeros_like(delta_r_obs_3d)
significance_map_fdr_positive_3d[significant_fdr_mask_3d & positive_mask] = delta_r_obs_3d[significant_fdr_mask_3d & positive_mask]

# Print summary of significant voxels
num_significant_voxels = np.sum(significant_fdr_mask_3d)
print(f"Number of significant voxels (FDR-corrected, p < 0.05, all): {num_significant_voxels}")
print(f"Proportion of significant voxels within brain mask (all): {num_significant_voxels / np.sum(brain_mask):.2%}")
num_significant_positive_voxels = np.sum(significant_fdr_mask_3d & positive_mask)
print(f"Number of significant voxels with positive delta: {num_significant_positive_voxels}")
print(f"Proportion of significant voxels with positive delta: {num_significant_positive_voxels / np.sum(brain_mask):.2%}")

# Create and save NIfTI files
significance_fdr_nifti = nib.Nifti1Image(significance_map_fdr_3d, affine)


nib.save(significance_fdr_nifti, 'results/significant_improvements_fdr_corrected_all.nii')
significance_fdr_positive_nifti = nib.Nifti1Image(significance_map_fdr_positive_3d, affine)
nib.save(significance_fdr_positive_nifti, 'results/significant_improvements_fdr_corrected_positive.nii')

# Plot glass brain (all significant voxels)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    significance_fdr_nifti,
    threshold=0,
    title='Significant ΔR² (Text+Audio vs Max(Text, Audio))\nFDR-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()

# Plot stat map (all significant voxels)
min_delta = np.min(np.abs(significance_map_fdr_3d[significance_map_fdr_3d != 0])) if np.any(significance_map_fdr_3d != 0) else 0
plotting.plot_stat_map(
    significance_fdr_nifti,
    bg_img=mni_template,
    threshold=min_delta,
    title='Significant ΔR² (Text+Audio vs Max(Text, Audio))\nFDR-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r',
    vmax=np.max(np.abs(significance_map_fdr_3d)) * 1.1 if np.any(significance_map_fdr_3d != 0) else 0.1,
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/significant_delta_r_fdr_stat_map_all_2mm.png', dpi=300)
plt.close()

# Plot glass brain (positive significant voxels)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    significance_fdr_positive_nifti,
    threshold=0,
    title='Significant Positive ΔR² (Text+Audio > Max(Text, Audio))\nFDR-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()
plt.savefig('results/significant_improvements_fdr_glass_brain_positive.png', dpi=300)
plt.close()

# Plot stat map (positive significant voxels)
min_delta_pos = np.min(significance_map_fdr_positive_3d[significance_map_fdr_positive_3d > 0]) if np.any(significance_map_fdr_positive_3d > 0) else 0
plotting.plot_stat_map(
    significance_fdr_positive_nifti,
    bg_img=mni_template,
    threshold=min_delta_pos,
    title='Significant Positive ΔR² (Text+Audio > Max(Text, Audio))\nFDR-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r',
    vmax=np.max(significance_map_fdr_positive_3d) * 1.1 if np.any(significance_map_fdr_positive_3d > 0) else 0.1,
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/significant_delta_r_fdr_stat_map_positive_2mm.png', dpi=300)
plt.close()

# Plot histogram of ΔR²
plt.figure(figsize=(8, 5))
plt.hist(delta_r_obs_flat, bins=100, color='steelblue', alpha=0.7)
plt.xlabel('ΔR² (Text+Audio - Max(Text, Audio))')
plt.ylabel('Voxel Count')
plt.title('Distribution of ΔR² within Brain Mask')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/delta_r_histogram_all_voxels.png')
plt.close()

# Diagnostic: Proportion of positive voxels
print(f"Proportion of voxels with ΔR² > 0: {np.sum(delta_r_obs_3d[brain_mask] > 0) / np.sum(brain_mask):.2%}")

# Diagnostic: Unthresholded ΔR² map
delta_r_nifti = nib.Nifti1Image(delta_r_obs_3d, affine)
plotting.plot_stat_map(
    delta_r_nifti,
    bg_img=mni_template,
    title='Unthresholded ΔR² (Text+Audio - Max(Text, Audio))',
    colorbar=True,
    cmap='RdBu_r',
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/unthresholded_delta_r_stat_map.png', dpi=300)
plt.close()