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
import logging


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
                                help="Use PCA for embeddings with a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--use_umap", action="store_true",
                                help="Use UMAP for dimensionality reduction (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.60,
                            help="Explained variance threshold for PCA (default: 0.60).")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony", "prosody", "semantic", "tom"],
                            help="List of tasks to include (default: all available tasks).")
    dataset_group.add_argument("--data_type", type=str, choices=["mc", "normalized", "unormalized", "normalized_time"], default="unormalized",
                            help="Type of fMRI data to use: mc (mean-centered), normalized, normalized_time, or unormalized (default: unormalized).")
    
    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--num_jobs", type=int, default=-1,
                                help="Number of parallel jobs for voxel processing (default: -1 for all cores).")
    
    logging.basicConfig(level=logging.INFO)

    return parser.parse_args()

def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    # Print settings, including num_jobs
    print(f"Running with settings:\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use audio_opensmile: {args.use_audio_opensmile}\n"
          f"- Use text_weighted: {args.use_text_weighted}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- Use UMAP: {args.use_umap}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Data type: {args.data_type}\n"
          f"- Number of jobs: {args.num_jobs}\n"
          f"- Included tasks: {', '.join(args.include_tasks)}")

    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])
    #participant_list = participant_list[0:5]
        
    mask = nib.load("ROIs/ROIall_bin.nii")
    exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')

    stim_df, resp, ids_list = analysis_helpers.load_dataset(args, paths, participant_list, resampled_mask)
    
    alphas = np.logspace(-1, 3, 10)

    _, corrs, valphas, fold_corrs = ridge_cv_participant(
        stim_df=stim_df, 
        resp=resp, 
        alphas=alphas, 
        participant_ids=ids_list, 
        nboots=len(participant_list),
        corrmin=-1, 
        singcutoff=1e-10, 
        normalpha=True, 
        use_corr=True, 
        return_wt=False,
        normalize_stim=False, 
        normalize_resp=True, 
        n_jobs=5, 
        with_replacement=False,
        bootstrap_n_jobs=args.num_jobs
    )
    
    # Prepare mask for mapping results to 3D brain space
    mask_data = resampled_mask.get_fdata()
    mask_bool = (mask_data > 0).flatten()
    volume_shape = mask_data.shape
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
    feature_str = "_".join(features_used) if features_used else "nofeatures"
    
    # Save corrs (mean correlations across folds)
    corrs_flat = np.zeros(np.prod(volume_shape))
    corrs_flat[mask_bool] = corrs
    corrs_3D = corrs_flat.reshape(volume_shape)
    corrs_nifti = nib.Nifti1Image(corrs_3D, affine=resampled_mask.affine)
    result_file_mean = os.path.join(paths["results_path"][args.data_type], f"correlation_map_mean_{feature_str}.npy")
    np.save(result_file_mean, corrs_3D)
    result_file_mean_nii = os.path.join(paths["results_path"][args.data_type], f"correlation_map_mean_{feature_str}.nii.gz")
    nib.save(corrs_nifti, result_file_mean_nii)
    
    # Save valphas (optimal alpha per voxel)
    valphas_flat = np.zeros(np.prod(volume_shape))
    valphas_flat[mask_bool] = valphas
    valphas_3D = valphas_flat.reshape(volume_shape)
    valphas_nifti = nib.Nifti1Image(valphas_3D, affine=resampled_mask.affine)
    result_file_valphas = os.path.join(paths["results_path"][args.data_type], f"valphas_map_{feature_str}.npy")
    np.save(result_file_valphas, valphas_3D)
    result_file_valphas_nii = os.path.join(paths["results_path"][args.data_type], f"valphas_map_{feature_str}.nii.gz")
    nib.save(valphas_nifti, result_file_valphas_nii)
    
    # Save fold_corrs (correlations per fold, averaged across folds for 3D mapping)
    # Note: fold_corrs shape is (M, 5), where M is number of voxels and 5 is number of folds
    # For NIfTI, save the mean across folds to maintain 3D shape; save full fold_corrs as .npy
    fold_corrs_mean = np.mean(fold_corrs, axis=1)  # Mean across folds
    fold_corrs_flat = np.zeros(np.prod(volume_shape))
    fold_corrs_flat[mask_bool] = fold_corrs_mean
    fold_corrs_3D = fold_corrs_flat.reshape(volume_shape)
    fold_corrs_nifti = nib.Nifti1Image(fold_corrs_3D, affine=resampled_mask.affine)
    result_file_fold_corrs = os.path.join(paths["results_path"][args.data_type], f"fold_corrs_mean_map_{feature_str}.npy")
    np.save(result_file_fold_corrs, fold_corrs_3D)
    result_file_fold_corrs_nii = os.path.join(paths["results_path"][args.data_type], f"fold_corrs_mean_map_{feature_str}.nii.gz")
    nib.save(fold_corrs_nifti, result_file_fold_corrs_nii)
    
    # Save full fold_corrs array (M, 5) as .npy for detailed analysis
    result_file_fold_corrs_full = os.path.join(paths["results_path"][args.data_type], f"fold_corrs_full_{feature_str}.npy")
    np.save(result_file_fold_corrs_full, fold_corrs)
    
    end_time = time.time()
    print("Total r2: %d" % sum(corrs * np.abs(corrs)))
    print(f"Analysis completed in {(end_time - start_time) / 60:.2f} minutes.")

# Run the script
if __name__ == "__main__":
    main()