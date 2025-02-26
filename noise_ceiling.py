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
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import analysis_helpers


def set_data(base_data):
        final_data = []

        for item in base_data:
            fmri_value = None
            participant = item["participant"]
            context = item["context_condition"]
            semantic = item["statement_condition"][:2]  # First two characters
            prosody = item["statement_condition"][-3:]
            task = item["task"]
            evaluation = item["evaluation"]
            situation = item["situation"]

            final_data.append({
                "context": context, "semantic": semantic, "prosody": prosody, 
                "task": task, "evaluation": evaluation, "fmri_value": fmri_value, "situation": situation, "participant": participant,

            })  

        df = pd.DataFrame(final_data)
        df.reset_index(drop=True, inplace=True)
        return df

def get_voxel_values(voxel, base_data, data):
    voxel_values = []        
    for item in base_data:
        voxel_values.append(item["fmri_data"][voxel])
    data["fmri_value"] = voxel_values
    return data

def noise_ceiling_data(data):
    participant_dfs = {}
    for participant in data["participant"].unique():
        participant_data = data[data["participant"] == participant].copy()
        other_participants_data = data[data["participant"] != participant]
        avg_other_fmri = other_participants_data.groupby(
            ["prosody", "context", "semantic", "task", "situation"]
        )["fmri_value"].mean().reset_index()
        merged_data = participant_data.merge(
            avg_other_fmri,
            on=["prosody", "context", "semantic", "task", "situation"],
            suffixes=("", "_others_avg"),
            how="left"
        )
        participant_dfs[participant] = merged_data
    return participant_dfs


def voxel_analysis(voxel, base_data, data, alpha):
    data_voxel = get_voxel_values(voxel, base_data, data)
    participant_data = noise_ceiling_data(data_voxel)
    participant_correlations = []

    for participant, data in participant_data.items():
        X = data["fmri_value_others_avg"].values.reshape(-1, 1)
        y = data["fmri_value"].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_correlations = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            correlation, _ = pearsonr(y_test, y_pred)
            fold_correlations.append(correlation)

        participant_correlations.append(np.mean(fold_correlations))
    return np.mean(participant_correlations)  


args = argparse.Namespace(
    img_size=[75, 92, 77],
    use_audio = True,
    use_text = False,
    use_context = False,
    use_base_features=True,
    use_pca=False, num_jobs = 15, alpha = 1, pca_threshold = 0.5)



paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])
database = analysis_helpers.load_dataset(args, paths, participant_list)
base_data = database.base_data
data = set_data(base_data)

voxel_list = list(np.ndindex(tuple(args.img_size)))
top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
top_voxels = analysis_helpers.get_top_voxels(database, args.img_size, voxel_list, top_voxels_path)


for voxel in top_voxels:
    print(f"Current voxel : {voxel}")
    alphas = [0.01, 0.1, 1, 10, 100]
    best_alpha = None
    best_correlation = -np.inf
    
    for alpha in alphas:
        corr = voxel_analysis(voxel, base_data, data, alpha)
        print(f"Alpha: {alpha}, Mean Correlation: {corr:.4f}")
        
        if corr > best_correlation:
            best_correlation = corr
            best_alpha = alpha
    
    print(f"Best Alpha: {best_alpha}, Highest Correlation: {best_correlation:.4f}")

results = Parallel(n_jobs=args.num_job)(
    delayed(voxel_analysis)(voxel, base_data, data, args.alpha) for voxel in voxel_list
)

    
    



    


