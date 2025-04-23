import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from scipy.ndimage import label
import os

# Load correlation maps
r_audio = np.load("results/r2_map_mean_audio_base_iro_sar.npy")
r_text = np.load("results/r2_map_mean_text_weighted_base_iro_sar.npy")
r_text_audio = np.load("results/r2_map_mean_audio_text_weighted_base_iro_sar.npy")
#brain_mask = np.logical_and.reduce([r_audio != 0, r_text != 0, r_text_audio != 0])

# Compute mean affine
data_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted"
all_affines = []
for subject in os.listdir(data_folder):
    subject_path = os.path.join(data_folder, subject)
    for file in os.listdir(subject_path):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            file_path = os.path.join(subject_path, file)
            nifti_img = nib.load(file_path)
            all_affines.append(nifti_img.affine)

affine = np.mean(all_affines, axis=0)
#affine = all_affines[2]

# Create NIfTI images
r_audio_nifti = nib.Nifti1Image(r_audio, affine)
r_text_nifti = nib.Nifti1Image(r_text, affine)
r_text_audio_nifti = nib.Nifti1Image(r_text_audio, affine)

# test = np.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results_test\correlation_map_mean_text_weighted_base_iro.npy")
# test_nifti = nib.Nifti1Image(test, affine)


# # Plot glass brain (all significant clusters)
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# plotting.plot_glass_brain(
#     test_nifti,
#     threshold=0,
#     title='Significant Delta R² (Text+Audio vs Max(Text,Audio))\nFWER-corrected p < 0.05',
#     colorbar=True,
#     plot_abs=False,
#     display_mode='ortho',
#     axes=ax
# )
# plt.tight_layout()

# Load and resample MNI152 brain mask
mni_brain_mask = datasets.load_mni152_brain_mask(resolution=2)
mni_template = datasets.load_mni152_template(resolution=2)
mni_brain_mask_resampled = image.resample_to_img(mni_brain_mask, r_audio_nifti, interpolation='nearest')
brain_mask = mni_brain_mask_resampled.get_fdata() > 0

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

# Step 6: Apply cluster-forming threshold
p_values_3d = np.zeros((X, Y, Z))
p_values_3d[brain_mask] = p_values
initial_mask_3d = p_values_3d < 0.01

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
significance_map_3d = np.zeros_like(delta_r_obs_3d)
significance_map_3d[clustered_mask_3d] = delta_r_obs_3d[clustered_mask_3d]

# Step 11 (Optional): Filter for positive improvements
positive_mask = delta_r_obs_3d > 0
significance_map_positive_3d = np.zeros_like(delta_r_obs_3d)
significance_map_positive_3d[clustered_mask_3d & positive_mask] = delta_r_obs_3d[clustered_mask_3d & positive_mask]

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
#nib.save(significance_nifti, 'results/significant_improvements_cluster_corrected_all.nii')

# Save NIfTI for positive clusters only
significance_positive_nifti = nib.Nifti1Image(significance_map_positive_3d, affine)
#nib.save(significance_positive_nifti, 'results/significant_improvements_cluster_corrected_positive.nii')

# Plot glass brain (all significant clusters)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    significance_nifti,
    threshold=0,
    title='Significant Delta R² (Text+Audio vs Max(Text,Audio))\nFWER-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()
#plt.savefig('results/significant_improvements_glass_brain_all.png', dpi=300)

# Plot stat map (all significant clusters)
min_delta = np.min(np.abs(significance_map_3d[significance_map_3d != 0])) if np.any(significance_map_3d != 0) else 0
plotting.plot_stat_map(
    significance_nifti,
    bg_img=mni_template,
    threshold=min_delta,
    title='Significant Delta R² (Text+Audio vs Max(Text, Audio))\nCluster-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r',
    vmax=np.max(np.abs(significance_map_3d)) * 1.1 if np.any(significance_map_3d != 0) else 0.1,
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/significant_delta_r_stat_map_all_2mm.png', dpi=300)
plt.close()

# Plot glass brain (positive clusters only)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    significance_positive_nifti,
    threshold=0,
    title='Significant Positive Delta R² (Text+Audio > Max(Text,Audio))\nFWER-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()
plt.savefig('results/significant_improvements_glass_brain_positive.png', dpi=300)
plt.close()

# Plot stat map (positive clusters only)
min_delta_pos = np.min(significance_map_positive_3d[significance_map_positive_3d > 0]) if np.any(significance_map_positive_3d > 0) else 0
plotting.plot_stat_map(
    significance_positive_nifti,
    bg_img=mni_template,
    threshold=min_delta_pos,
    title='Significant Positive Delta R² (Text+Audio > Max(Text, Audio))\nCluster-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r',
    vmax=np.max(significance_map_positive_3d) * 1.1 if np.any(significance_map_positive_3d > 0) else 0.1,
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/significant_delta_r_stat_map_positive_2mm.png', dpi=300)
plt.close()

# Plot histogram of delta_r_obs_flat
plt.figure(figsize=(8, 5))
plt.hist(delta_r_obs_flat, bins=100, color='steelblue', alpha=0.7)
plt.xlabel('Delta R² (Text+Audio - Max(Text, Audio))')
plt.ylabel('Voxel Count')
plt.title('Distribution of ΔR² within Brain Mask')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/delta_r_histogram_all_voxels.png')
plt.close()

# Diagnostic: Proportion of positive voxels
print(f"Proportion of voxels with delta_r_obs_3d > 0: {np.sum(delta_r_obs_3d[brain_mask] > 0) / np.sum(brain_mask):.2%}")

# Diagnostic: Unthresholded delta_r_obs_3d
delta_r_nifti = nib.Nifti1Image(delta_r_obs_3d, affine)
plotting.plot_stat_map(
    delta_r_nifti,
    bg_img=mni_template,
    title='Unthresholded Delta R² (Text+Audio - Max(Text, Audio))',
    colorbar=True,
    cmap='RdBu_r',
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/unthresholded_delta_r_stat_map.png', dpi=300)
plt.close()