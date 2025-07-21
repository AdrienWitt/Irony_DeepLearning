import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs, math_img, iter_img
from nilearn.masking import compute_epi_mask
from nilearn.image import resample_to_img
import pandas as pd
from nilearn.glm.first_level import compute_regressor

# Define paths
folder_fmri = r'D:\Preproc_Analyses\data_done'
folder_audio = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\Stimuli'
files_type = ['wrMF']
output_dir_fmri = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\normalized_time_unsmoothed'
if not os.path.exists(output_dir_fmri):
    os.makedirs(output_dir_fmri)
output_dir_mask = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\group_masks'

group_mask_dir = os.path.join(output_dir_mask, 'group_mask')
if not os.path.exists(group_mask_dir):
    os.makedirs(group_mask_dir)

# fMRI parameters
TR = 0.65  # Repetition time in seconds

def select_files(root_folder, files_type):
    participant_folders = glob.glob(os.path.join(root_folder, 'p*'))
    participant_files = {}
    for participant_folder in participant_folders:
        participant = participant_folder[-3:]
        run_folders = glob.glob(os.path.join(participant_folder, 'RUN*'))
        run_files = {}
        for run_folder in run_folders:
            run = run_folder[-4:]
            nii_files = glob.glob(os.path.join(run_folder, f'{files_type}*.nii'))
            run_files[run] = nii_files
            participant_files[participant] = run_files
    return participant_files

def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile_p')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t')
        key = file_name[-8:-4].upper()
        dfs[key] = df
    return dfs

def crop_skull_background(fmri, participant, run_number, output_dir, group_mask=None):
    if group_mask is not None:
        # Use provided group mask
        mask = group_mask
        mask_filename = None
    else:
        # Check for existing individual mask in output_dir_mask/{participant}/
        mask_dir = os.path.join(output_dir_mask, participant)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        mask_filename = os.path.join(mask_dir, f"{participant}_{run_number}_mask.nii.gz")
        if os.path.exists(mask_filename):
            print(f"Loading existing individual mask for {participant}_{run_number}: {mask_filename}")
            mask = nib.load(mask_filename)
        else:
            # Generate and save new individual mask
            print(f"Generating new individual mask for {participant}_{run_number}")
            mask = compute_epi_mask(fmri)
            nib.save(mask, mask_filename)
    
    print(f"Mask shape for {participant}_{run_number}: {mask.shape}")
    masked_list = []
    for img in iter_img(fmri):
        masked = math_img('img*mask', img=img, mask=mask)
        masked_list.append(masked)
    masked_concat = concat_imgs(masked_list)
    return masked_concat, mask_filename

def mean_z_norm(fmri):
    nonzero_mask = fmri != 0
    nonzero_voxels = fmri[nonzero_mask]
    if len(nonzero_voxels) == 0:
        print("Warning: No nonzero voxels found!")
        return fmri
    global_mean = np.mean(nonzero_voxels)
    global_std = np.std(nonzero_voxels)
    print(f"Mean (excluding background): {global_mean}")
    print(f"Std (excluding background): {global_std}")
    fmri_temp = np.zeros_like(fmri)
    fmri_temp[nonzero_mask] = (fmri[nonzero_mask] - global_mean) / global_std
    return fmri_temp

def mean_center(fmri):
    """Center non-zero voxels by subtracting their mean, preserving variance."""
    nonzero_mask = fmri != 0
    nonzero_voxels = fmri[nonzero_mask]
    if len(nonzero_voxels) == 0:
        print("Warning: No nonzero voxels found!")
        return fmri
    global_mean = np.mean(nonzero_voxels)
    print(f"Mean (excluding background): {global_mean}")
    fmri_temp = fmri.copy()
    fmri_temp[nonzero_mask] -= global_mean
    return fmri_temp

def mean_z_norm_time_inmask(fmri):
    # Create a mask for non-zero voxels across all time points
    nonzero_mask = np.any(fmri != 0, axis=3)  # Shape: (x, y, z)
    if not np.any(nonzero_mask):
        print("Warning: No nonzero voxels found!")
        return fmri
    # Extract non-zero voxel time series
    nonzero_voxels = fmri[nonzero_mask, :]  # Shape: (n_nonzero_voxels, time)
    # Compute mean and std along time axis (axis=1 in nonzero_voxels)
    voxel_means = np.mean(nonzero_voxels, axis=1, keepdims=True)  # Shape: (n_nonzero_voxels, 1)
    voxel_stds = np.std(nonzero_voxels, axis=1, keepdims=True)   # Shape: (n_nonzero_voxels, 1)
    # Avoid division by zero
    voxel_stds = np.where(voxel_stds == 0, 1, voxel_stds)
    # Initialize output array
    fmri_temp = fmri.copy()
    # Compute z-scores for non-zero voxels
    fmri_temp[nonzero_mask, :] = (nonzero_voxels - voxel_means) / voxel_stds
    print(f"Mean range across non-zero voxels: {np.min(voxel_means):.4f} to {np.max(voxel_means):.4f}")
    print(f"Std range across non-zero voxels: {np.min(voxel_stds):.4f} to {np.max(voxel_stds):.4f}")
    return fmri_temp

def mean_z_norm_time(fmri):
    fmri_temp = np.zeros_like(fmri, dtype=np.float64)
    mean_time = np.mean(fmri, axis=3, keepdims=True)
    std_time = np.std(fmri, axis=3, keepdims=True)
    std_time = np.where(std_time == 0, 1, std_time)
    fmri_temp = (fmri - mean_time) / std_time
    return fmri_temp
    
def mean_center_time(fmri):
    # Create a mask for non-zero voxels across all time points
    nonzero_mask = np.any(fmri != 0, axis=3)  # Shape: (x, y, z)   
    if not np.any(nonzero_mask):
        print("Warning: No nonzero voxels found!")
        return fmri
    # Extract non-zero voxel time series
    nonzero_voxels = fmri[nonzero_mask, :]  # Shape: (n_nonzero_voxels, time)
    # Compute mean along time axis (axis=1 in nonzero_voxels)
    voxel_means = np.mean(nonzero_voxels, axis=1, keepdims=True)  # Shape: (n_nonzero_voxels, 1)
    # Initialize output array
    fmri_temp = fmri.copy()
    # Subtract mean for non-zero voxels
    fmri_temp[nonzero_mask, :] = nonzero_voxels - voxel_means
    print(f"Mean range across non-zero voxels: {np.min(voxel_means):.4f} to {np.max(voxel_means):.4f}")
    return fmri_temp


def compute_group_mask_inter(mask_files, output_dir):
    if not mask_files:
        raise ValueError("No mask files found to compute group mask.")
    masks = [nib.load(f) for f in mask_files]
    # Compute intersection (all voxels must be non-zero in all masks)
    group_mask_data = np.prod([img.get_fdata() for img in masks], axis=0)
    group_mask = nib.Nifti1Image(group_mask_data, masks[0].affine, masks[0].header)
    group_mask_filename = os.path.join(output_dir, "group_mask_inter.nii.gz")
    nib.save(group_mask, group_mask_filename)
    print(f"Group mask saved to {group_mask_filename}")
    return group_mask_filename

def compute_group_mask_threshold(mask_files, output_dir, threshold=0.25):
    if not mask_files:
        raise ValueError("No mask files found to compute group mask.")
    # Load masks
    masks = [nib.load(f) for f in mask_files]    
    mask_sum = np.sum([img.get_fdata() for img in masks], axis=0)
    # Compute thresholded group mask (voxels where overlap >= threshold * num_masks)
    num_masks = len(mask_files)
    group_mask_data = (mask_sum >= threshold * num_masks).astype(np.int16)    
    group_mask = nib.Nifti1Image(group_mask_data, masks[10].affine, masks[10].header)    
    group_mask_filename = os.path.join(output_dir, f"group_mask_threshold_{threshold}.nii.gz")
    nib.save(group_mask, group_mask_filename)
    print(f"Group mask saved to {group_mask_filename}")
    
    return group_mask_filename


# Step 1: Collect or generate individual masks
mask_files = []
for file_type in files_type:
    participant_files = select_files(folder_fmri, file_type)
    for participant, runs in participant_files.items():
        dfs = load_dataframe(os.path.join(folder_fmri, participant))
        for run_number, run_files in runs.items():
            subj_dir = os.path.join(output_dir_mask, participant)
            if not os.path.exists(subj_dir):
                os.makedirs(subj_dir)
            mask_filename = os.path.join(subj_dir, f"{participant}_{run_number}_mask.nii.gz")
            
            # Check if mask already exists
            if os.path.exists(mask_filename):
                print(f"Loading existing mask for {participant}_{run_number}: {mask_filename}")
                mask_files.append(mask_filename)
            else:
                # Generate new mask
                print(f"Generating new mask for {participant}_{run_number}")
                concatenated_img = concat_imgs(run_files)
                _, mask_filename = crop_skull_background(concatenated_img, participant, run_number, subj_dir)
                mask_files.append(mask_filename)

# Step 2: Compute group mask
group_mask_filename = compute_group_mask_threshold(mask_files, group_mask_dir)
group_mask = nib.load(group_mask_filename)

# Step 3: Reprocess fMRI data with group mask
for file_type in files_type:
    participant_files = select_files(folder_fmri, file_type)
    for participant, runs in participant_files.items():
        dfs = load_dataframe(os.path.join(folder_fmri, participant))
        for run_number, run_files in runs.items():
            # Load and preprocess fMRI data with group mask
            concatenated_img = concat_imgs(run_files)
            subj_dir = os.path.join(output_dir_fmri, participant)
            if not os.path.exists(subj_dir):
                os.makedirs(subj_dir)
            #cropped_img, mask_filename = crop_skull_background(concatenated_img, participant, run_number, subj_dir)
            fmri = concatenated_img.get_fdata()
            affine = concatenated_img.affine
            header = concatenated_img.header
            #fmri_normalized = mean_center(fmri)
            #fmri_normalized = fmri
            #fmri_normalized = mean_z_norm(fmri)
            fmri_normalized = mean_z_norm_time(fmri)

            # Get corresponding dataframe
            df = dfs[run_number]
            df = df.rename(columns=lambda x: x.strip())

            # Define frame times
            frame_times = np.arange(0, fmri_normalized.shape[-1] * TR, TR)
                        
            # Process each event
            for i, row in df.iterrows():
                context = row['Context']
                statement = row['Statement']
                task = row['task']
                jitter = row['Jitter']
                start_statement = row['Real_Time_Onset_Statement']
                end_statement = row['Real_Time_End_Statement']
                duration_statement = end_statement - start_statement
                end_evaluation = row['Real_Time_End_Evaluation']
                
                if np.isnan(end_evaluation):
                    end_evaluation = row['Real_Time_Onset_Evaluation'] + 5
                else:
                    end_evaluation = row['Real_Time_End_Evaluation']

                onsets = [start_statement]
                durations = [duration_statement]
                amplitudes = [1.0]
                
                exp_condition_s = [np.array(onsets), np.array(durations), np.array(amplitudes)]
                
                # Generate HRF-convolved regressor
                hrf_regressor, _ = compute_regressor(
                    exp_condition=exp_condition_s,
                    hrf_model='glover',
                    frame_times=frame_times,
                    oversampling=16
                )

                # Define time window
                start_scan = round(start_statement / TR)
                end_scan = round(end_evaluation / TR)
                
                start_scan = max(0, min(start_scan, fmri_normalized.shape[-1] - 1))
                end_scan = max(start_scan, min(end_scan, fmri_normalized.shape[-1]))

                # Extract scans and HRF weights
                scans = fmri_normalized[..., start_scan:end_scan]
                hrf_weights = hrf_regressor[start_scan:end_scan, 0]
                hrf_weights = hrf_weights / np.sum(hrf_weights)

                # Compute HRF-weighted average
                if scans.shape[-1] > 0 and len(hrf_weights) == scans.shape[-1]:
                    weighted_scans = np.average(scans, axis=-1, weights=hrf_weights)
                else:
                    print(f"Skipping {participant}_{task}_{context[:-4]}_{statement[:-4]}: Mismatch in scan and weight dimensions")
                    continue

                # Save fMRI data
                file = nib.Nifti1Image(weighted_scans, affine, header)
                filename = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement'
                nib.save(file, os.path.join(subj_dir, filename + ".nii.gz"))

print("Processing complete.")