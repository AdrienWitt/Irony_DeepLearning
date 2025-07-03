import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image 
from scipy.ndimage import label
import os
from statsmodels.stats.multitest import fdrcorrection
from nilearn.image import resample_to_img
import analysis_helpers


# Load correlation maps
r_text_audio = np.load("results_5/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text = np.load("results_5/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_5/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")


# # Load correlation maps
# r_text_audio = np.load("results_irosar/normalized_time/folds_correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
# r_text = np.load("results_irosar/normalized_time/folds_correlation_map_flat_text_weighted_base_5.npy")
# r_audio = np.load("results_irosar/normalized_time/folds_correlation_map_flat_audio_opensmile_base_5.npy")


r_text_audio = np.load("results_irosar/normalized_time/correlation_map_flat_audio_opensmile_text_weighted_base_5.npy")
r_text = np.load("results_irosar/normalized_time/correlation_map_flat_text_weighted_base_5.npy")
r_audio = np.load("results_irosar/normalized_time/correlation_map_flat_audio_opensmile_base_5.npy")


delta_r = r_text_audio - np.maximum(r_text, r_audio)
# delta_r_mean = delta_r.mean(axis = 1)
print(f"delta_r shape: {delta_r.shape}")
print(f"Max delta_r: {delta_r.max()}")

# --------------------------------------------
# 3) Load mask and reshape to 3D
# --------------------------------------------

mask = nib.load("ROIs/ROIall_bin.nii")
example_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
resampled_mask = resample_to_img(mask, example_data, interpolation='nearest')

mask_data = resampled_mask.get_fdata()
mask_bool = mask_data > 0

volume_shape = mask_data.shape

# Fill a 3D volume
delta_r_flat = np.zeros(mask_bool.size)
delta_r_flat[mask_bool.flatten()] = delta_r
delta_r_3D = delta_r_flat.reshape(volume_shape)
threshold = np.percentile(delta_r, 90)
above_thresh_3D = delta_r_3D.copy()
above_thresh_3D[above_thresh_3D <= threshold] = 0 

results_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results_5\normalized_time"

corrs_nifti = nib.Nifti1Image(above_thresh_3D, affine=resampled_mask.affine)
nib.save(corrs_nifti, os.path.join(results_path, "delta_r_all_thresh.nii"))




print(f"3D shape: {delta_r_3D.shape}")

# --------------------------------------------
# 4) Apply a cluster-defining threshold
# --------------------------------------------

threshold = np.percentile(delta_r, 90)
above_thresh = delta_r_3D > threshold

structure = np.ones((3, 3, 3))  # 26-connectivity

cluster_labels, n_clusters = label(above_thresh, structure=structure)

print(f"Found {n_clusters} clusters above threshold")

# Compute cluster sizes
cluster_sizes = []
for c in range(1, n_clusters + 1):
    cluster_sizes.append(np.sum(cluster_labels == c))

print(f"Cluster sizes: {cluster_sizes}")

# --------------------------------------------------------
# 5) Permutation test: cluster size
# --------------------------------------------------------

rng = np.random.default_rng(seed=42)
n_permutations = 1000
max_cluster_sizes = []

for p in range(n_permutations):
    # Randomly flip signs voxel-wise
    perm_signs = rng.choice([-1, 1], size=delta_r.shape)
    perm_delta_r = delta_r * perm_signs

    # Put permuted data into 3D
    perm_flat = np.zeros(mask_bool.size)
    perm_flat[mask_bool.flatten()] = perm_delta_r
    perm_3D = perm_flat.reshape(volume_shape)

    # Apply same threshold
    perm_above = perm_3D > threshold

    # Label clusters
    perm_labels, perm_n_clusters = label(perm_above, structure=structure)

    # Get cluster sizes
    perm_sizes = [np.sum(perm_labels == c) for c in range(1, perm_n_clusters + 1)]

    if perm_sizes:
        max_cluster_sizes.append(max(perm_sizes))
    else:
        max_cluster_sizes.append(0)

print(f"Example null max cluster sizes: {max_cluster_sizes[:5]}")

# --------------------------------------------------------
# 6) Compute cluster-level p-values (size)
# --------------------------------------------------------

max_cluster_sizes = np.array(max_cluster_sizes)

p_values = []
for real_size in cluster_sizes:
    p = np.mean(max_cluster_sizes >= real_size)
    p_values.append(p)

print(f"Cluster p-values (size): {p_values}")

# --------------------------------------------
# 6) Compute cluster-level p-values
# --------------------------------------------

max_cluster_masses = np.array(max_cluster_masses)

p_values = []
for real_mass in cluster_masses:
    p = np.mean(max_cluster_masses >= real_mass)
    p_values.append(p)

print(f"Cluster p-values: {p_values}")

results_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results_irosar\normalized_time"

mask = nib.load("ROIs/ROIall_bin.nii")
exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')
mask_data = resampled_mask.get_fdata()
mask_bool = (mask_data > 0).flatten()
volume_shape = mask_data.shape
corrs_flat = np.zeros(np.prod(volume_shape))
corrs_flat[mask_bool] = delta_r
corrs_3D = corrs_flat.reshape(volume_shape)
corrs_nifti = nib.Nifti1Image(corrs_3D, affine=resampled_mask.affine)
nib.save(corrs_nifti, os.path.join(results_path, "delta_r.nii"))




from scipy.stats import wilcoxon, ttest_1samp

# Voxel-wise Wilcoxon
n_voxels, n_folds = delta_r.shape
p_values = np.ones(n_voxels)

for v in range(n_voxels):
    # Only test if there's variation
    if np.all(delta_r[v] == 0):
        continue
    try:
        w, p = wilcoxon(delta_r[v], alternative='two-sided')
        p_values[v] = p
    except ValueError:
        # Wilcoxon might fail if all folds are identical
        p_values[v] = 1.0

# FDR correct
from statsmodels.stats.multitest import fdrcorrection
significant, pvals_fdr = fdrcorrection(p_values, alpha=0.05)

# Keep mean delta only for significant voxels
delta_r_mean = delta_r.mean(axis=1)
keep_voxels = significant & (delta_r_mean > 0)
delta_r_sig = np.zeros_like(delta_r_mean)
delta_r_sig[keep_voxels] = delta_r_mean[keep_voxels]

n_sig_voxels = np.sum(keep_voxels)
print(f"Number of significant positive voxels: {n_sig_voxels}")

##########################################################################################


import numpy as np
from tqdm import tqdm  # for progress bar

n_voxels, n_folds = delta_r.shape
n_permutations = 5000

# Actual mean per voxel
delta_r_mean = delta_r.mean(axis=1)

# Null distribution: shape (voxels, permutations)
null_dist = np.zeros((n_voxels, n_permutations))

rng = np.random.default_rng(42)  # Reproducibility

print("Running permutations...")
for p in tqdm(range(n_permutations)):
    # Random sign flips: shape (1, folds)
    signs = rng.choice([-1, 1], size=(1, n_folds))
    # Broadcast multiply: flips signs for all voxels
    permuted = delta_r * signs
    # Mean across folds
    null_dist[:, p] = permuted.mean(axis=1)

# Two-sided p-values:
# p = proportion of null means >= |obs|
abs_null = np.abs(null_dist)
abs_obs = np.abs(delta_r_mean[:, np.newaxis])

p_values = np.mean(abs_null >= abs_obs, axis=1)


from statsmodels.stats.multitest import fdrcorrection

significant, pvals_fdr = fdrcorrection(p_values, alpha=0.05)

keep_voxels = significant & (delta_r_mean > 0)
delta_r_sig = np.zeros_like(delta_r_mean)
delta_r_sig[keep_voxels] = delta_r_mean[keep_voxels]

n_sig_voxels = np.sum(keep_voxels)
print(f"Number of significant positive voxels: {n_sig_voxels}")


results_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results_5\normalized_time"

mask = nib.load("ROIs/ROIall_bin.nii")
exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')
mask_data = resampled_mask.get_fdata()
mask_bool = (mask_data > 0).flatten()
volume_shape = mask_data.shape
corrs_flat = np.zeros(np.prod(volume_shape))
corrs_flat[mask_bool] = delta_r
corrs_3D = corrs_flat.reshape(volume_shape)
corrs_nifti = nib.Nifti1Image(corrs_3D, affine=resampled_mask.affine)
nib.save(corrs_nifti, os.path.join(results_path, "delta_r.nii"))


