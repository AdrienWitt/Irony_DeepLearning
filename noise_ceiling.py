# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:11:51 2025

@author: adywi
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import dataset  # Assuming this is a custom module

# Function to set up paths dynamically
def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": os.path.join(base_path, "data", "fmri"),
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text", "statements"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "results_path": os.path.join(base_path, "results"),
    }

    # Create results directory if it doesn't exist
    os.makedirs(paths["results_path"], exist_ok=True)
    
    return paths

# Function to load dataset and split participants
def load_datasets(paths, img_size, mode):
    participant_list = os.listdir(paths["data_path"])
    train_participants, test_participants = train_test_split(participant_list, test_size=0.5, random_state=42)

    database_train = dataset.BaseDataset(
        participant_list=train_participants,
        data_path=paths["data_path"],
        fmri_data_path=paths["fmri_data_path"],
        img_size=img_size,
        embeddings_text_path=paths["embeddings_text_path"],
        embeddings_audio_path=paths["embeddings_audio_path"],
        mode=mode
    )

    database_test = dataset.BaseDataset(
        participant_list=test_participants,
        data_path=paths["data_path"],
        fmri_data_path=paths["fmri_data_path"],
        img_size=img_size,
        embeddings_text_path=paths["embeddings_text_path"],
        embeddings_audio_path=paths["embeddings_audio_path"],
        mode="audio_only" if mode == "base_features" else "base_features"
    )

    return database_train, database_test

def compute_correlation(y_1, y_2):
    """
    Compute Pearson correlation between two arrays.
    """
    if np.std(y_1) > 0 and np.std(y_2) > 0:
        return pearsonr(y_1, y_2)[0]
    else:
        return 0  # Return 0 if there's no variance in either array


def get_voxel_values(voxel, base_data):
    voxel_values = []
    filtered_data_list = []  # Stores dictionaries without "fmri_data"

    for item in base_data:
        voxel_values.append(item["fmri_data"][voxel])  # Extract voxel value
        filtered_data = {k: v for k, v in item.items() if k != "fmri_data"}  
        filtered_data_list.append(filtered_data)  

    # Convert filtered data to DataFrame
    df = pd.DataFrame(filtered_data_list)
    df["voxel_value"] = voxel_values  # Add voxel values as a new column
    
    return df

# Main function
img_size = (75, 92, 77)
mode = "base_features"
voxel_list = list(np.ndindex(img_size))  
paths = get_paths()
database_train, database_test = load_datasets(paths, img_size, mode)
base_data1 = database_train.base_data
a = base_data1[1:4]
filtered_data = {k: v for k, v in a.items() if k not in "fmri_data"}
voxel = voxel_list[222222]
base_data = get_voxel_values(voxel, base_data1)


noise_map = np.zeros(img_size)
for voxel in voxel_list:
    df_1 = database_train.get_voxel_values(voxel)
    df_2 = database_test.get_voxel_values(voxel)
    y_1 = df_1["fmri_value"].values
    y_2 = df_2["fmri_value"].values
    
result_file = os.path.join(paths["results_path"], f"noise_ceiling.npy")

    


