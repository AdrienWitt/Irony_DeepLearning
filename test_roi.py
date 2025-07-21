# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 17:08:31 2025

@author: adywi
"""
from nilearn import datasets, image, plotting
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt



mask = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\ROIs\ROIall.nii")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    mask,
    threshold=0,
    title='Significant Delta R² (Text+Audio vs Max(Text,Audio))\nFWER-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()


# # Load correlation maps
r_audio = np.load("results/normalized/correlation_map_mean_audio_opensmile_base_sar_iro_pro_sem_tom.npy")
r_text = np.load("results/normalized/correlation_map_mean_text_weighted_base_sar_iro_pro_sem_tom.npy")
r_text_audio = np.load("results/normalized/correlation_map_mean_audio_opensmile_text_weighted_base_sar_iro_pro_sem_tom.npy")
brain_mask = nib.load(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\group_masks\group_mask\group_mask_threshold_0.25.nii.gz")
affine = brain_mask.affine

valid_mask = (r_text != 0) & (r_audio != 0) & (r_text_audio != 0)
delta_r = np.zeros_like(r_text)
delta_r[valid_mask] = r_text[valid_mask] - np.maximum(r_text_audio[valid_mask], r_audio[valid_mask])


delta_r_nifti = nib.Nifti1Image(delta_r, affine)

# ------------------------------------
# Step 1: Fetch Harvard-Oxford Atlas & build IFG mask
# ------------------------------------
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
atlas_img = atlas.maps  # proper load
atlas_data = atlas_img.get_fdata()
labels = atlas.labels

# Choose IFG subregions
target_labels = [
    "Inferior Frontal Gyrus, pars opercularis",
    "Inferior Frontal Gyrus, pars triangularis"
]
label_indices = [labels.index(name) for name in target_labels]

# Binary mask for those labels
ifg_mask_data = np.isin(atlas_data, label_indices).astype(np.uint8)
ifg_mask_img = nib.Nifti1Image(ifg_mask_data, atlas_img.affine, atlas_img.header)

# ------------------------------------
# Step 2: Resample IFG mask to match delta_r_nifti map
# ------------------------------------
# Load your delta_r map (already computed earlier)
# If saved: delta_r_nifti = nib.load("your_delta_r_map.nii.gz")

resampled_ifg_mask = image.resample_to_img(ifg_mask_img, delta_r_nifti, interpolation='nearest')

# ------------------------------------
# Step 3: Apply IFG mask to delta_r_nifti map
# ------------------------------------
mask_data = resampled_ifg_mask.get_fdata()

# Ensure binary mask
mask_data = (mask_data > 0).astype(np.uint8)

# Apply mask (retain only IFG voxels)
masked_delta_r_data = delta_r * mask_data

# Save new masked image
masked_delta_r_nifti = nib.Nifti1Image(masked_delta_r_data, delta_r_nifti.affine, delta_r_nifti.header)
#nib.save(masked_delta_r_nifti, "masked_delta_r_left_ifg.nii.gz")
print("✅ Masked delta_r map saved as masked_delta_r_left_ifg.nii.gz")

# ------------------------------------
# Step 4: (Optional) Plot masked map
# ------------------------------------
# Plot glass brain (all significant clusters)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
plotting.plot_glass_brain(
    mask,
    threshold=0,
    title='Significant Delta R² (Text+Audio vs Max(Text,Audio))\nFWER-corrected p < 0.05',
    colorbar=True,
    plot_abs=False,
    display_mode='ortho',
    axes=ax
)
plt.tight_layout()

##################################################
import argparse
import dataset
import analysis_helpers
import os

paths = analysis_helpers.get_paths()


args = argparse.Namespace(
    use_audio = False,
    use_text = False,
    use_base_features=True,
    use_text_weighted = True,
    use_audio_opensmile = True,
    include_tasks = ["irony", "sarcasm", "semantic", "prosody", 'tom'],
    use_pca=True, num_jobs = 1, alpha = 0.1, pca_threshold = 0.5, use_umap = False, data_type = 'normalized_time')

participant_list = os.listdir(paths["data_path"])

participant_list = participant_list[0:5]

def load_dataset(args, paths, participant_list):
    """Loads the dataset using parsed arguments."""

    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"][args.data_type],  # Select the correct path based on data type
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "embeddings_audio_opensmile_path": paths["embeddings_audio_opensmile_path"],
        "use_base_features": args.use_base_features,
        "use_text": args.use_text,
        "use_audio": args.use_audio,
        "use_audio_opensmile": args.use_audio_opensmile,
        "use_text_weighted": args.use_text_weighted,
        "pca_threshold": args.pca_threshold,
        "use_pca" : args.use_pca,
        "use_umap" : args.use_umap,
        "included_tasks": args.include_tasks
    }

    data, data_fmri, ids_list = dataset.WholeBrainDataset(participant_list=participant_list, **dataset_args).create_data()
    
    return data, data_fmri, ids_list

stim_df, resp, ids_list = load_dataset(args, paths, participant_list)


resp = resp[:, 150:200]

from ridge_cv import ridge_cv_participant
import numpy as np 

alphas = np.logspace(1, 4, 10)


_, corrs, valphas, fold_corrs = ridge_cv_participant(stim_df, resp, alphas, ids_list, nboots=5,
                         corrmin=0.1, singcutoff=1e-10, normalpha=True, use_corr=True, return_wt=False,
                         normalize_stim=False, normalize_resp=True, n_jobs=1, with_replacement=False,
                         bootstrap_n_jobs=1)



a = nib.load(r'C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\normalized_time\p01\p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz')

import os
import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def flatten_nifti(input_path, output_path):
    """Flatten a 3D NIfTI file to 1 x voxels and save as a numpy array."""
    try:
        logging.info(f"Processing {input_path}")
        
        # Verify input file exists
        if not input_path.exists():
            logging.error(f"Input file does not exist: {input_path}")
            return
        
        # Load NIfTI file
        img = nib.load(str(input_path))
        data = img.get_fdata()
        
        # Log data shape
        logging.info(f"Loaded {input_path} with shape {data.shape}")
        
        # Ensure data is 3D (expected 79x95x79)
        if data.ndim != 3:
            logging.error(f"Skipping {input_path}: Expected 3D data, got {data.ndim}D with shape {data.shape}")
            return
        if data.shape != (79, 95, 79):
            logging.warning(f"Unexpected shape for {input_path}: Got {data.shape}, expected (79, 95, 79)")
        
        # Flatten to 1 x voxels (1 x 592,895)
        flattened = data.reshape(1, -1)
        logging.info(f"Flattened shape: {flattened.shape}")
        
        # Verify output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save flattened array
        np.save(output_path, flattened)
        logging.info(f"Saved flattened data to {output_path}")
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")

def process_fmri_files(input_dir, output_dir, num_workers=None):
    """Process all NIfTI files in input_dir in parallel and save to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    if not input_dir.is_dir():
        logging.error(f"Input path is not a directory: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Collect all NIfTI files and their output paths
    file_pairs = []
    for root, _, files in os.walk(input_dir):
        rel_path = Path(root).relative_to(input_dir)
        out_subdir = output_dir / rel_path
        out_subdir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                input_file = Path(root) / file
                base_name = Path(file).stem
                if base_name.endswith('.nii'):
                    base_name = Path(base_name).stem
                output_file = out_subdir / f"{base_name}.npy"
                file_pairs.append((input_file, output_file))
    
    if not file_pairs:
        logging.warning(f"No NIfTI files found in {input_dir}")
        return
    
    logging.info(f"Found {len(file_pairs)} NIfTI files to process")
    
    # Use number of CPU cores if num_workers is not specified
    logging.info(f"Using {num_workers} worker(s)")
    
    # Process files (serial if num_workers=1, parallel otherwise)

    for input_file, output_file in file_pairs:
        flatten_nifti(input_file, output_file)

    
    logging.info(f"Processed {len(file_pairs)} files")

if __name__ == "__main__":
    try:
        # Example usage
        input_directory = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\normalized_time_unsmoothed"
        output_directory = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\normalized_time_unsmoothed_flatten"
        process_fmri_files(input_directory, output_directory, num_workers=20)
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
    
    
    
for file in os.listdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\normalized_time\p51"):
    img = nib.load(file)
    print(img.shape)
    
    
import os
from pathlib import Path

def rename_file(input_path, new_filename):
    """Rename a file by removing '_masked' from its filename."""
    try:
        # Get the directory of the input file
        input_dir = input_path.parent
        # Create the new file path
        output_path = input_dir / new_filename
        # Rename the file
        input_path.rename(output_path)
        print(f"Renamed {input_path} to {output_path}")
    except Exception as e:
        print(f"Error renaming {input_path}: {str(e)}")

def process_fmri_files(input_dir):
    """Process all NIfTI files in input_dir, removing '_masked' from filenames."""
    input_dir = Path(input_dir)
    
    # Walk through all subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.nii', '.nii.gz', '.npy')):
                input_file = Path(root) / file
                # Remove '_masked' from the filename
                new_filename = file.replace('_masked', '')
                if new_filename != file:  # Only rename if '_masked' was found
                    rename_file(input_file, new_filename)

if __name__ == "__main__":
    # Example usage
    input_directory = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\normalized_time_flatten"  # Replace with your input path
    process_fmri_files(input_directory)