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
r_text_audio = np.load("results_50/normalized_time/folds_correlation_map_flat_audio_opensmile_text_weighted_base.npy")
r_text = np.load("results_50/normalized_time/folds_correlation_map_flat_text_weighted_base.npy")
r_audio = np.load("results_50/normalized_time/folds_correlation_map_flat_audio_opensmile_base.npy")


delta_r = r_text_audio - np.maximum(r_text, r_audio)


results_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\results_50\normalized_time"


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