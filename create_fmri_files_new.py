# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:01:04 2024

@author: adywi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:43:16 2024

@author: adywi
"""

import os
import glob
import numpy as np
import nibabel as nib
from nilearn.image import concat_imgs
import pandas as pd
from nilearn.image import crop_img, mean_img, iter_img, math_img
import matplotlib.pyplot as plt
from nilearn.masking import compute_epi_mask
import torch
from pydub import AudioSegment

folder_fmri = r'D:\Preproc_Analyses\data_done'
folder_audio = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\Stimuli'
files_type = ['wrMF']
output_dir_fmri = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri'
output_dir_audio = r'C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\audio'

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
                run_files[run]  = nii_files
                participant_files[participant] = run_files
    return participant_files
           
def save_concatenated(file_type, root_folder, output_dir):
    participant_files = select_files()
    for participant, runs in participant_files.items():
        for run in runs.values():
            concatenated_img = concat_imgs(run['files'])
            output_filename = f'{file_type}_{participant}_{run}'
            output_dir = os.path.join(root_folder, output_dir)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, output_filename)
            nib.save(concatenated_img, output_path)
            
def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t') 
        key = file_name[-8:-4]
        dfs[key] = df
    return dfs

def crop_skull_background(fmri):
    mask = compute_epi_mask(fmri)
    masked_list = []
    for img in iter_img(fmri):
            masked = math_img('img*mask', img = img, mask=mask)
            masked_list.append(masked)
    masked_concat = concat_imgs(masked_list)
    croped_img = crop_img(masked_concat)
    return croped_img

def mean_z_norm(fmri):
    global_mean = fmri.mean()
    print("mean:", global_mean)
    global_std = fmri.std()
    print("std:", global_std)
    fmri_temp = (fmri - global_mean) / global_std
    return fmri_temp

def audiofile_merger(context, statement, jitter):
    context_path = os.path.join(folder_audio, context)
    statement_path = os.path.join(folder_audio, statement)
    context_audio = AudioSegment.from_file(context_path)
    statement_audio = AudioSegment.from_file(statement_path)
    jitter_audio = AudioSegment.silent(duration=jitter * 1000)
    merged_audio = context_audio + jitter_audio + statement_audio
    return merged_audio
    
for file_type in files_type:
    participant_files = select_files(folder_fmri, file_type)  
    for participant, runs in participant_files.items():
        dfs = load_dataframe(os.path.join(folder_fmri, participant))       
        for run_number, run, df in zip(runs.keys(), runs.values(), dfs.values()): 
            concatenated_img = concat_imgs(run)
            croped_img = crop_skull_background(concatenated_img)
            fmri = croped_img.get_fdata()
            affine = croped_img.affine 
            header = croped_img.header
            fmri_normalized = mean_z_norm(fmri)
            df = df.rename(columns=lambda x: x.strip())
            for i, row in df.iterrows():
               context = row['Context']
               statement = row['Statement']
               task = row['task']
               jitter = row['Jitter']
               time_context = row['Time_Context']               
               time_statement = row['Time_Statement']

               
               ## Fmri

               start_context = row['Real_Time_Onset_Context']
               end_context = row['Real_Time_End_Context']
               
               if time_context < 0.65:
                   start_scan_context = round(start_context / 0.65)
                   end_scan_context = start_scan_context + 1
               else:
                   start_scan_context = round(start_context / 0.65)
                   end_scan_context = round(end_context / 0.65)
                   
               scans_context = fmri_normalized[..., start_scan_context:end_scan_context].mean(axis = 3)
               file_context = nib.Nifti1Image(scans_context, affine, header)
               filename_context = f'{participant}_{task}_{i}_{context[:-4]}'
               
               start_statement = row['Real_Time_Onset_Statement']
               end_statement = row['Real_Time_End_Statement']
               
               if time_statement < 0.65:
                   start_scan_statement = round(start_statement / 0.65)
                   end_scan_statement = start_scan_statement + 1
               else:
                   start_scan_statement = round(start_statement / 0.65)
                   end_scan_statement = round(end_statement / 0.65)
               
               scans_statement = fmri_normalized[..., start_scan_statement:end_scan_statement].mean(axis = 3)
               file_statement = nib.Nifti1Image(scans_statement, affine, header)
               filename_statement = f'{participant}_{task}_{i}_{statement[:-4]}'
               
               start_jitter = row['Real_Time_End_Context']
               end_jitter = row['Real_Time_Onset_Statement']
               
               if jitter < 0.65:
                   start_scan_jitter = round(start_jitter / 0.65)
                   end_scan_jitter = start_scan_jitter + 1
               else:
                   start_scan_jitter = round(start_jitter / 0.65)
                   end_scan_jitter = round(end_jitter / 0.65)
               
               scans_jitter = fmri_normalized[..., start_scan_jitter:end_scan_jitter].mean(axis = 3)
               file_jitter = nib.Nifti1Image(scans_jitter, affine, header)
               filename_jitter = f'{participant}_{task}_{i}_jitter'
               #################################################################
               end_evaluation = row['Real_Time_End_Evaluation']
               end_scan_evaluation = round(end_evaluation / 0.65)

               scans_stat_eval = fmri_normalized[..., start_scan_statement:end_scan_evaluation].mean(axis = 3)
               file_stat_eval = nib.Nifti1Image(scans_stat_eval, affine, header)
               filename_stat_eval = f'{participant}_{task}_{i}_{statement[:-4]}_eval'

               
               subj_dir = os.path.join(output_dir_fmri, participant)
               if not os.path.exists(subj_dir):
                   os.makedirs(subj_dir)
               #nib.save(file_context, os.path.join(subj_dir,filename_context +".nii.gz"))
               #nib.save(file_statement, os.path.join(subj_dir,filename_statement +".nii.gz"))
               #nib.save(file_jitter, os.path.join(subj_dir,filename_jitter +".nii.gz"))
               nib.save(file_stat_eval, os.path.join(subj_dir,filename_stat_eval +".nii.gz"))

               
               
     

     