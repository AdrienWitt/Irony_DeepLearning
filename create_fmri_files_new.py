import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs
import pandas as pd
from nilearn.image import crop_img, iter_img, math_img
from nilearn.masking import compute_epi_mask
from nilearn.glm.first_level import compute_regressor
from pydub import AudioSegment

# Define paths
folder_fmri = r'D:\Preproc_Analyses\data_done'
folder_audio = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\Stimuli'
files_type = ['swrMF']
output_dir_fmri = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri\weighted'
   
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

def crop_skull_background(fmri):
    mask = compute_epi_mask(fmri)
    print(mask.shape)
    masked_list = []
    for img in iter_img(fmri):
        masked = math_img('img*mask', img=img, mask=mask)
        masked_list.append(masked)
    masked_concat = concat_imgs(masked_list)
    return masked_concat

def mean_z_norm(fmri):
    global_mean = fmri.mean()
    print(f"Mean: {global_mean}")
    global_std = fmri.std()
    print(f"Std: {global_std}")
    fmri_temp = (fmri - global_mean) / global_std
    return fmri_temp

# Main processing loop
for file_type in files_type:
    participant_files = select_files(folder_fmri, file_type)
    for participant, runs in participant_files.items():
        dfs = load_dataframe(os.path.join(folder_fmri, participant))  # Adjusted to match participant folder naming
        for run_number, run_files in runs.items():
            # Load and preprocess fMRI data
            concatenated_img = concat_imgs(run_files)
            cropped_img = crop_skull_background(concatenated_img)
            fmri = cropped_img.get_fdata()
            affine = cropped_img.affine
            header = cropped_img.header
            fmri_normalized = mean_z_norm(fmri)

            # Get corresponding dataframe
            df = dfs[run_number]
            df = df.rename(columns=lambda x: x.strip())

            # Define frame times for the entire run
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

                onsets = [start_statement]  # List with one onset
                durations = [duration_statement]  # List with one duration
                amplitudes = [1.0]
                
                exp_condition_s = [np.array(onsets), np.array(durations), np.array(amplitudes)]
                
                # Generate HRF-convolved regressor
                hrf_regressor, _ = compute_regressor(
                    exp_condition=exp_condition_s,
                    hrf_model='glover',  # Canonical SPM HRF
                    frame_times=frame_times,
                    oversampling=16
                )

                # Define time window for fMRI extraction (statement + HRF tail)
                start_scan = round(start_statement / TR)
                # end_scan = round(end_statement / TR) + round(HRF_TAIL / TR)
                end_scan = round(end_evaluation / TR)
                
                start_scan = max(0, min(start_scan, fmri_normalized.shape[-1] - 1))
                end_scan = max(start_scan, min(end_scan, fmri_normalized.shape[-1]))

                # Extract scans and HRF weights
                scans = fmri_normalized[..., start_scan:end_scan]
                hrf_weights = hrf_regressor[start_scan:end_scan, 0]

                # Compute HRF-weighted average
                if scans.shape[-1] > 0 and len(hrf_weights) == scans.shape[-1]:
                    weighted_scans = np.average(scans, axis=-1, weights=hrf_weights)
                else:
                    print(f"Skipping {participant}_{task}_{context[:-4]}_{statement[:-4]}: Mismatch in scan and weight dimensions")
                    continue

                # Save fMRI data
                file = nib.Nifti1Image(weighted_scans, affine, header)
                filename = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement_weighted'
                subj_dir = os.path.join(output_dir_fmri, participant)
                if not os.path.exists(subj_dir):
                    os.makedirs(subj_dir)
                nib.save(file, os.path.join(subj_dir, filename + ".nii.gz"))

print("Processing complete.")