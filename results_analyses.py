import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from statsmodels.stats.multitest import fdrcorrection

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

# Step 6: Compute p-values for each voxel
p_values_flat = np.zeros(V)
for v in range(V):
    if delta_r_obs_flat[v] > 0:  # Only compute for positive delta_r
        p_values_flat[v] = np.mean(delta_r_perm[v, :] >= delta_r_obs_flat[v])
    else:
        p_values_flat[v] = 1.0  # Non-positive delta_r gets p=1

# Step 7: Apply FDR correction
rejected, p_values_fdr = fdrcorrection(p_values_flat, alpha=0.05, method='indep')

# Step 8: Create significance map (p < 0.05, FDR-corrected)
significance_map_3d = np.zeros_like(delta_r_obs_3d)
significant_voxels = np.zeros_like(brain_mask, dtype=bool)
significant_voxels[brain_mask] = rejected  # Map rejected hypotheses back to brain voxels
significant_voxels = significant_voxels & positive_mask  # Restrict to positive delta_r
significance_map_3d[significant_voxels] = delta_r_obs_3d[significant_voxels]

# Print summary
if np.any(significant_voxels):
    print(f"Number of significant voxels (FDR, p < 0.05): {np.sum(significant_voxels)}")
else:
    print("No significant voxels found (FDR, p < 0.05).")

# Create and save NIfTI
significance_nifti = nib.Nifti1Image(significance_map_3d, affine)
nib.save(significance_nifti, 'results/significant_improvements_fdr_corrected_masked.nii')


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
    title='Significant Delta R² (Text+Audio > Max(Text, Audio))\nFDR-corrected p < 0.05',
    colorbar=True,
    cmap='RdBu_r',
    vmax=np.max(significance_map_3d) * 1.1 if np.any(significance_map_3d > 0) else 1.0,
    display_mode='ortho',
    cut_coords=(0, 0, 0)
)
plt.savefig('results/significant_delta_r_stat_map_fdr_2mm.png', dpi=300)
plt.close()