import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from scipy.ndimage import label, binary_dilation
from scipy.ndimage import sum as ndi_sum

def compute_tfce(stat_img, mask=None, E=0.5, H=2.0, dh=0.1):
    """
    Compute Threshold-Free Cluster Enhancement (TFCE) for a 3D statistic image.
    
    Parameters:
    - stat_img: 3D numpy array of statistic values (e.g., delta_r).
    - mask: 3D boolean array, same shape as stat_img, to restrict computation.
    - E: Extent exponent (default: 0.5).
    - H: Height exponent (default: 2.0).
    - dh: Height increment for integration (default: 0.1).
    
    Returns:
    - tfce_img: 3D array of TFCE scores, same shape as stat_img.
    """
    if mask is None:
        mask = np.ones_like(stat_img, dtype=bool)
    stat_img = stat_img * mask  # Apply mask
    tfce_img = np.zeros_like(stat_img)
    max_stat = np.max(stat_img[mask])
    if max_stat <= 0:
        return tfce_img  # No positive stats
    
    # Compute TFCE by integrating over thresholds
    thresholds = np.arange(0, max_stat + dh, dh)
    for h in thresholds:
        # Create binary map at threshold h
        cluster_map = stat_img > h
        cluster_map = cluster_map * mask
        # Label connected components
        labeled_array, _ = label(cluster_map, structure=np.ones((3, 3, 3)))
        # Compute cluster sizes
        cluster_sizes = np.bincount(labeled_array.ravel())[1:] if np.max(labeled_array) > 0 else []
        # Assign TFCE contribution to each voxel
        for cluster_id in range(1, len(cluster_sizes) + 1):
            cluster_mask = labeled_array == cluster_id
            extent = cluster_sizes[cluster_id - 1] ** E
            height = h ** H
            tfce_img[cluster_mask] += extent * height * dh
    
    return tfce_img

# Load MNI template and brain mask
mni_template = datasets.load_mni152_template(resolution=2)
affine = mni_template.affine
brain_mask_nifti = datasets.load_mni152_brain_mask(resolution=2)
brain_mask = brain_mask_nifti.get_fdata().astype(bool)

# Load correlation maps
r_audio = np.load("results/correlation_map_mean_audio_base.npy")
r_text = np.load("results/correlation_map_mean_text_weighted_base.npy")
r_text_audio = np.load("results/correlation_map_mean_audio_text_weighted_base.npy")

# Check shapes
print("Correlation map shape:", r_audio.shape)
print("MNI template shape:", mni_template.shape)
shapes_match = r_audio.shape == mni_template.shape

# Create NIfTI images
r_audio_nifti = nib.Nifti1Image(r_audio, affine)
r_text_nifti = nib.Nifti1Image(r_text, affine)
r_text_audio_nifti = nib.Nifti1Image(r_text_audio, affine)

# Resample only if needed
if not shapes_match:
    print("Resampling to MNI 2 mm space...")
    r_audio_nifti = image.resample_img(r_audio_nifti, target_affine=affine, target_shape=mni_template.shape)
    r_text_nifti = image.resample_img(r_text_nifti, target_affine=affine, target_shape=mni_template.shape)
    r_text_audio_nifti = image.resample_img(r_text_audio_nifti, target_affine=affine, target_shape=mni_template.shape)
    r_audio = r_audio_nifti.get_fdata()
    r_text = r_text_nifti.get_fdata()
    r_text_audio = r_text_audio_nifti.get_fdata()
else:
    print("Skipping resampling: shapes match.")

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

# Step 3: Filter for positive delta_r
positive_mask = delta_r_obs_3d > 0
delta_r_positive_3d = delta_r_obs_3d * positive_mask

# Step 4: Flatten only brain voxels
brain_mask_flat = brain_mask.ravel()
valid_voxels = brain_mask_flat
V = np.sum(valid_voxels)  # Number of brain voxels
r_audio_flat = r_audio.reshape(-1)[valid_voxels]
r_text_flat = r_text.reshape(-1)[valid_voxels]
r_text_audio_flat = r_text_audio.reshape(-1)[valid_voxels]
delta_r_obs_flat = delta_r_obs_3d.reshape(-1)[valid_voxels]
positive_mask_flat = positive_mask.reshape(-1)[valid_voxels]

# Step 5: Permutation test
num_permutations = 5000
delta_r_perm = np.zeros((V, num_permutations))
for i in range(num_permutations):
    permuted_indices = np.random.permutation(V)
    r_audio_perm = r_audio_flat[permuted_indices]
    r_text_perm = r_text_flat[permuted_indices]
    r_text_audio_perm = r_text_audio_flat[permuted_indices]
    delta_r_perm[:, i] = r_text_audio_perm - np.maximum(r_audio_perm, r_text_perm)

# Step 6: Compute TFCE for observed data
tfce_obs_3d = compute_tfce(delta_r_positive_3d, mask=brain_mask)

# Step 7: Compute TFCE for permutations
max_tfce_perm = np.zeros(num_permutations)
for i in range(num_permutations):
    delta_r_perm_3d = np.zeros((X, Y, Z))
    delta_r_perm_3d[brain_mask] = delta_r_perm[:, i]
    delta_r_perm_3d = delta_r_perm_3d * positive_mask  # Only positive delta_r
    tfce_perm_3d = compute_tfce(delta_r_perm_3d, mask=brain_mask)
    max_tfce_perm[i] = np.max(tfce_perm_3d[brain_mask]) if np.any(tfce_perm_3d[brain_mask]) else 0

# Step 8: Compute TFCE p-values
tfce_p_values_3d = np.zeros((X, Y, Z))
# Get flat indices of brain mask
brain_mask_indices = np.where(brain_mask.ravel())[0]
# Compute p-values for each voxel in the brain mask
tfce_p_values = np.zeros(np.sum(brain_mask))
for idx, flat_idx in enumerate(brain_mask_indices):
    # Convert flat index to 3D coordinates
    i, j, k = np.unravel_index(flat_idx, (X, Y, Z))
    tfce_p_values[idx] = np.mean(max_tfce_perm >= tfce_obs_3d[i, j, k])
tfce_p_values_3d[brain_mask] = tfce_p_values

# Step 9: Create significance map (p < 0.05, corrected)
significance_map_3d = np.zeros_like(delta_r_obs_3d)
significant_voxels = (tfce_p_values_3d < 0.05) & positive_mask
significance_map_3d[significant_voxels] = delta_r_obs_3d[significant_voxels]

# Print summary
if np.any(significant_voxels):
    print(f"Number of significant voxels (TFCE, p < 0.05): {np.sum(significant_voxels)}")
else:
    print("No significant voxels found (TFCE, p < 0.05).")

# Create and save NIfTI
significance_nifti = nib.Nifti1Image(significance_map_3d, affine)
nib.save(significance_nifti, 'results/significant_improvements_tfce_corrected_masked.nii')

# Plot using nilearn
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(significance_nifti, threshold=0, 
                         title='Significant improvement (text+audio > max(text,audio))\nFWER-corrected p < 0.05',
                         colorbar=True, plot_abs=False,
                         display_mode='ortho', axes=ax)

# Plot results
min_delta = np.min(significance_map_3d[significance_map_3d > 0]) if np.any(significance_map_3d > 0) else 0
plotting.plot_stat_map(
    significance_nifti,
    bg_img=mni_template,
    threshold=min_delta,
    title='Significant Delta R² (Text+Audio > Max(Text, Audio))\nTFCE-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r',
    vmax=np.max(significance_map_3d) * 1.1,
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/significant_delta_r_stat_map_tfce_2mm.png', dpi=300)
plt.close()