# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:08:58 2024

@author: adywi
"""

import os
import pandas as pd
from pydub import AudioSegment

# Define the directory paths
data_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\data"
stimuli_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\Stimuli"
output_dir = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Brain_DeepLearning\Stimuli_merged"

# Function to merge audio files with a jitter
def merge_audio_files(context_file, statement_file, jitter, output_file):
    # Load the audio files
    context_audio = AudioSegment.from_file(context_file)
    statement_audio = AudioSegment.from_file(statement_file)
    
    # Create a silent audio segment for the jitter
    jitter_audio = AudioSegment.silent(duration=jitter * 1000)
    
    # Concatenate the context, jitter, and statement
    merged_audio = context_audio + jitter_audio + statement_audio
    
    # Export the merged audio
    merged_audio.export(output_file, format="wav")

# Iterate through each participant folder
for participant in os.listdir(data_folder):
    participant_path = os.path.join(data_folder, participant)
    
    if os.path.isdir(participant_path):
        # Iterate through each result file
        for file in os.listdir(participant_path):
            if file.startswith("Resultfile") and file.endswith(".txt"):
                file_path = os.path.join(participant_path, file)
                
                # Load the result file into a DataFrame
                df = pd.read_csv(file_path, sep='\t') 
                
                # Iterate through each row in the DataFrame
                for index, row in df.iterrows():
                    context_stimulus = row[' Context ']
                    statement_stimulus = row[' Statement']
                    jitter = row[' Jitter']
                    run = row[' run']
                    
                    context_file = os.path.join(stimuli_folder, context_stimulus)
                    statement_file = os.path.join(stimuli_folder, statement_stimulus)
                    
                    # Define the output file path
                    output_file = os.path.join(output_dir, f"{participant}_{run}_{context_stimulus}_{statement_stimulus}")
                    
                    # Merge the audio files
                    merge_audio_files(context_file, statement_file, jitter, output_file)