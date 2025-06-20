# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 13:07:34 2025

@author: adywi
"""

import os
import time
import argparse
import numpy as np
import analysis_helpers
import nibabel as nib
from ridge_cv import ridge_cv_participant
from nilearn.image import resample_to_img

# os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
# args = argparse.Namespace(
#     use_audio = False,
#     use_text = False,
#     use_base_features=True,
#     use_text_weighted = True,
#     use_audio_opensmile = True,
#     include_tasks = ["irony", "sarcasm", "tom", "semantic", "prosody"],
#     use_pca=True, num_jobs = 1, alpha = 0.1, pca_threshold = 0.6, use_umap = False, data_type = 'normalized_time')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # **Dataset-related arguments**
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                                help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--use_text", action="store_true", 
                                help="Include text in dataset (default: False).")
    dataset_group.add_argument("--use_audio", action="store_true", 
                                help="Include audio in dataset (default: False).")
    dataset_group.add_argument("--use_audio_opensmile", action="store_true", 
                                help="Include openSMILE audio features in dataset (default: False).")
    dataset_group.add_argument("--use_text_weighted", action="store_true", 
                                help="Include text_weighted in dataset (default: False).")
    dataset_group.add_argument("--use_pca", action="store_true", 
                                help="Use PCA for embeddings with the a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--use_umap", action="store_true",
                                help="Use UMAP for dimensionality reduction (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.60,
                            help="Explained variance threshold for PCA (default: 0.60).")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony", "prosody", "semantic", "tom"],
                            help="List of tasks to include (default: all available tasks).")
    dataset_group.add_argument("--data_type", type=str, choices=["mc", "normalized", "unormalized", "normalized_time"], default="unormalized",
                            help="Type of fMRI data to use: mc (mean-centered), normalized, normalized_time, or unormalized (default: unormalized).")
    
def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    print(f"Running with settings:\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use audio_opensmile: {args.use_audio_opensmile}\n"
          f"- Use text_weighted: {args.use_text_weighted}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- Use UMAP: {args.use_umap}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Ridge alpha: {args.alpha}\n"
          f"- Number of parallel jobs: {args.num_jobs}\n"
          f"- Data type: {args.data_type}\n"
          f"- Included tasks: {', '.join(args.include_tasks)}")
    
    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])
    
    mask = nib.load("ROIs/ROIall_bin.nii")
    exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')

    
    stim_df, resp, ids_list = analysis_helpers.load_dataset(args, paths, participant_list, resampled_mask)
    
    alphas = np.logspace(-1, 3, 10)

    _, corrs, valphas, fold_corrs = ridge_cv_participant(stim_df, resp, alphas, ids_list, nboots=50,
                             corrmin=-1, singcutoff=1e-10, normalpha=True, use_corr=True, return_wt=False,
                             normalize_stim=False, normalize_resp=True, n_jobs=5, with_replacement=False,
                             bootstrap_n_jobs=50)
    
    mask_data = mask.get_fdata()
    mask_bool = (mask_data > 0).flatten()
    volume_shape = mask_data.shape
    corrs_flat = np.zeros(np.prod(volume_shape))
    corrs_flat[mask_bool] = corrs
    corrs_3D = corrs_flat.reshape(volume_shape)
    corrs_nifti = nib.Nifti1Image(corrs_3D, affine=resampled_mask.affine)
    
    features_used = []
    if args.use_text:
        features_used.append("text")
    if args.use_audio:
        features_used.append("audio")
    if args.use_audio_opensmile:
        features_used.append("audio_opensmile")
    if args.use_text_weighted:
        features_used.append("text_weighted")
    if args.use_base_features:
        features_used.append("base")
    # Create a feature string (e.g., "text_audio" if both are enabled)
    feature_str = "_".join(features_used) if features_used else "nofeatures"
    
    
    result_file_mean = os.path.join(paths["results_path"][args.data_type], f"correlation_map_mean_{feature_str}.npy")
    np.save(result_file_mean, corrs_nifti.get_fdata())
    result_file_mean_nii = os.path.join(paths["results_path"][args.data_type], f"correlation_map_mean_{feature_str}.nii.gz")
    nib.save(corrs_nifti, result_file_mean_nii)

    



    


    
    
    
    
    
    
    
