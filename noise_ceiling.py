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

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")
    
    parser.add_argument("--img_size", type=int, nargs=3, default=[75, 92, 77],
                        help="Size of fMRI images as three integers (default: 75 92 77).")
    parser.add_argument("--mode", type=str, choices=["base_features", "audio_only"], default="base_features",
                        help="Mode for dataset loading (default: base_features).")
    
    return parser.parse_args()

# Function to set up paths dynamically
def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": os.path.join(base_path, "data", "fmri"),
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text", "statements"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "noise_ceiling": os.path.join(base_path, "results"),
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


