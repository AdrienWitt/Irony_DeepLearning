import os
import argparse
import numpy as np
import analysis_helpers
import nibabel as nib
from nilearn.image import resample_to_img

from nilearn import datasets, image

args = argparse.Namespace(
    use_audio = False,
    use_text = False,
    use_base_features=True,
    use_text_weighted = True,
    use_audio_opensmile = True,
    include_tasks = ["irony", "sarcasm"],
    optimize_alpha = True,
    use_pca=False, num_jobs = 1, pca_threshold = 0.5, use_umap = False, data_type = 'normalized_time', 
    alpha_min = 1, alpha_max = 6, num_alphas = 20)


paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])
 
 
# mask = nib.load("ROIs/ROIall_bin.nii")
icbm = datasets.fetch_icbm152_2009()
mask_path = icbm['mask']
# Load the mask as a Nifti image object
mask = image.load_img(mask_path)

exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')

stim_df, resp, ids_list = analysis_helpers.load_dataset(args, paths, participant_list, resampled_mask)

# Après ton load_dataset, ajoute ceci :

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def check_actual_variance(stim_df, prefix, threshold):
    cols = [c for c in stim_df.columns if c.startswith(prefix)]
    if not cols:
        print(f"No columns found for prefix: {prefix}")
        return
    
    embeddings = stim_df[cols].to_numpy()
    
    # Refit PCA with threshold to get exact number of components kept
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca_full = PCA()
    pca_full.fit(embeddings_scaled)
    var_explained = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Exact number of components at thresholdthreshold
    n_components = np.searchsorted(var_explained, threshold) + 1
    actual_variance = var_explained[n_components - 1]
    
    print(f"\n{'='*50}")
    print(f"{prefix}")
    print(f"  Requested threshold : {threshold*100:.1f}%")
    print(f"  Components kept     : {n_components}")
    print(f"  Actual variance     : {actual_variance*100:.2f}%")
    print(f"  Difference          : {(actual_variance - threshold)*100:.2f}%")

check_actual_variance(stim_df, "emb_weighted_", threshold=0.80)
check_actual_variance(stim_df, "emb_audio_opensmile_", threshold=0.80)
