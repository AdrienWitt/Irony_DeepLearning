# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:33:40 2025

@author: adywi
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from scipy.stats import pearsonr
from joblib import Parallel, delayed
import os

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
# Load dataset
import dataset  

# Paths
data_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\behavioral"
fmri_data_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\data\fmri"
embeddings_text_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\statements"
embeddings_audio_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\audio"

# fMRI image size
img_size = (75, 92, 77)  
voxel_list = list(np.ndindex(img_size))  

# Split participants into train and test sets
participant_list = os.listdir(data_path)
train_participants, test_participants = train_test_split(participant_list, test_size=0.2, random_state=42)

# Load dataset for training and testing
database_train = dataset.BaseDataset(
    participant_list=train_participants,
    data_path=data_path,
    fmri_data_path=fmri_data_path,
    img_size=img_size,
    embeddings_text_path=embeddings_text_path, embeddings_audio_path=embeddings_audio_path, mode = "text_audio"
)


# Define file path to save top voxels
top_voxels_path = "top10_voxels.csv"

if os.path.exists(top_voxels_path):
    # Load top voxels from file
    df_voxels = pd.read_csv(top_voxels_path)
    top_voxels = [tuple(x) for x in df_voxels.to_records(index=False)]
    print(f"Loaded {len(top_voxels)} voxels from {top_voxels_path}")
else:
    # Compute mean activation per voxel
    mean_activation = {voxel: np.mean(database_train.get_voxel_values(voxel)["fmri_value"].values) for voxel in voxel_list}

    # Compute threshold for top 10% of voxels
    threshold = np.percentile(list(mean_activation.values()), 90)  # 90th percentile

    # Select voxels with mean activation above threshold
    top_voxels = [voxel for voxel, activation in mean_activation.items() if activation >= threshold]

    # Save the top voxels
    df_voxels = pd.DataFrame(top_voxels, columns=["X", "Y", "Z"])
    df_voxels.to_csv(top_voxels_path, index=False)

    print(f"Computed and saved {len(top_voxels)} top voxels to {top_voxels_path}")


alpha_values = [.001, .01, .1, 1, 5, 10, 15, 20] 


def voxel_analysis(voxel):
    """Train Ridge regression with CV, find best alpha, and compute correlation."""
    df_train = database_train.get_voxel_values(voxel)
    X_train = df_train.drop(columns=["fmri_value"]).values
    y_train = df_train["fmri_value"].values  

    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # GridSearchCV to tune alpha
    ridge = Ridge()
    param_grid = {'alpha': alpha_values}
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Get best model and alpha
    best_alpha = grid_search.best_params_['alpha']
    best_ridge = grid_search.best_estimator_

    return voxel, best_alpha

# Parallel processing for top 10 voxels
num_jobs = -1  # Use all CPU cores
results = Parallel(n_jobs=num_jobs)(delayed(voxel_analysis)(voxel) for voxel in top_voxels)

results_path = "best_alphas.csv"

df_results = pd.DataFrame(results, columns=["Voxel", "Best_Alpha", "Correlation"])
df_results.to_csv(results_path, index=False)
print(f"Best alpha values saved to {results_path}!")