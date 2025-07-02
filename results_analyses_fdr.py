import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, datasets, image
from scipy.ndimage import label
import os
from statsmodels.stats.multitest import fdrcorrection

# Load correlation maps
r_audio = np.load("results_50/normalized_time/")
r_text = np.load("results_50/normalized_time")
r_text_audio = np.load("results_50/normalized_time")


mask = nib.load("ROIs/ROIall_bin.nii")
exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")

resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')
mask_data = resampled_mask.get_fdata()
mask_bool = (mask_data > 0).flatten()
volume_shape = mask_data.shape
corrs_flat = np.zeros(np.prod(volume_shape))
corrs_flat[mask_bool] = corrs
corrs_3D = corrs_flat.reshape(volume_shape)
corrs_nifti = nib.Nifti1Image(corrs_3D, affine=resampled_mask.affine)
np.save(os.path.join(results_path, f"correlation_map_3D_{feature_str}.npy"), corrs_3D)
nib.save(corrs_nifti, os.path.join(results_path, f"correlation_map_{feature_str}.nii.gz"))