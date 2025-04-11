# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:23:20 2025

@author: adywi
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import analysis_helpers 
import os
import argparse

def get_voxel_values(voxel, base_data, data):
    voxel_values = []        
    for item in base_data:
        voxel_values.append(item["fmri_data"][voxel])
    data["fmri_value"] = voxel_values
    return data

def set_data(base_data):
    final_data = []
    for item in base_data:
        fmri_value = None
        participant = item["participant"]
        context = item["context_condition"]
        semantic = item["statement_condition"][:2]
        prosody = item["statement_condition"][-3:]
        task = item["task"]
        evaluation = item["evaluation"]
        situation = item["situation"]
        gender = item["gender"]  # Fixed comma here too
        age = item["age"]      # Removed comma

        final_data.append({
            "context": context, "semantic": semantic, "prosody": prosody, 
            "task": task, "evaluation": evaluation, "fmri_value": fmri_value, 
            "situation": situation, "participant": participant, "gender": gender,
            "age": age
        })  

    df = pd.DataFrame(final_data)
    df.reset_index(drop=True, inplace=True)
    return df


def voxel_noise_ceiling_stacked(voxel, database, alpha):
    base_data = database.base_data
    data = set_data(base_data)
    data_voxel = get_voxel_values(voxel, base_data, data)
    participant_correlations = []
    
    for participant in data_voxel["participant"].unique():
        participant_data = data_voxel[data_voxel["participant"] == participant].copy()
        other_participants_data = data_voxel[data_voxel["participant"] != participant]
        
        X_stacked = []
        y_stacked = []
        feature_data = []  # To store raw features before stacking
        
        for idx, row in participant_data.iterrows():
            condition_match = other_participants_data[
                (other_participants_data["prosody"] == row["prosody"]) &
                (other_participants_data["context"] == row["context"]) &
                (other_participants_data["semantic"] == row["semantic"]) &
                (other_participants_data["task"] == row["task"]) &
                (other_participants_data["situation"] == row["situation"])
            ]
            
            if len(condition_match) == 0:
                continue
            
            # Extract fMRI values from other participants
            X_fmri_others = condition_match["fmri_value"].values.reshape(-1, 1)
            
            # Keep raw features (not dummified yet)
            base_features_df = condition_match[['context', 'semantic', 'prosody', 'task', 'participant', 'evaluation', 'gender', 'age']].copy()
            
            # Scale evaluation (still numeric, so we can do this now)
            base_features_df['evaluation'] = base_features_df['evaluation'].fillna(base_features_df['evaluation'].median())
            if base_features_df['evaluation'].max() != base_features_df['evaluation'].min():
                base_features_df['evaluation'] = (base_features_df['evaluation'] - base_features_df['evaluation'].min()) / \
                                                (base_features_df['evaluation'].max() - base_features_df['evaluation'].min())
            
            # Scale age
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            base_features_df['age'] = scaler.fit_transform(base_features_df[['age']])
            
            # Store raw features and fMRI values together
            feature_data.append(base_features_df)
            X_stacked.append(X_fmri_others)
            y_row = np.full(len(X_fmri_others), row["fmri_value"])
            y_stacked.append(y_row)
        
        if not X_stacked:
            participant_correlations.append(0)
            continue
        
        # Stack fMRI values
        X_fmri_stacked = np.vstack(X_stacked)
        y = np.hstack(y_stacked)
        
        # Combine all feature data into one DataFrame
        all_features_df = pd.concat(feature_data, axis=0, ignore_index=True)
        
        # Now apply dummy encoding to the full stacked feature set
        categorical_cols = ['context', 'semantic', 'prosody', 'task', 'participant', 'gender']
        all_features_df = pd.get_dummies(all_features_df, columns=categorical_cols, drop_first=True, dtype=int)
        
        # Combine fMRI values with the dummified features
        X = np.hstack([X_fmri_stacked, all_features_df.values])
        
        # Filter out zero targets
        valid_idx = y != 0
        X, y = X[valid_idx], y[valid_idx]
        
        alpha = (alpha * X.shape[1])/79
        
        if len(X) < 5:
            participant_correlations.append(0)
            continue
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_corrs = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            corr = pearsonr(y_test, y_pred)[0] if np.std(y_test) > 0 else 0
            fold_corrs.append(corr)
        
        participant_correlations.append(np.mean(fold_corrs))
    
    mean_corr = np.mean(participant_correlations)
    return voxel, mean_corr


args = argparse.Namespace(
    img_size=[75, 92, 77],
    use_audio = False,
    use_text = False,
    use_text_weighted = False,
    use_context = False,
    use_base_features=True,
    use_umap = False,
    use_pca=False, num_jobs = 5, alpha = 0.01, pca_threshold = 0.5)

# Main execution (same args as above)
paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])
database = analysis_helpers.load_dataset(args, paths, participant_list)

img_size = (79, 95, 79)
voxel_list = list(np.ndindex(img_size))

print("Computing stacked noise ceiling...")
correlation_map = np.zeros(img_size)

results = Parallel(n_jobs=args.num_jobs, verbose=1)(
    delayed(voxel_noise_ceiling_stacked)(voxel, database, args.alpha)
    for voxel in voxel_list
)

for voxel, corr in results:
    correlation_map[voxel] = corr

output_file = os.path.join(paths["results_path"], "noise_ceiling_stacked_map.npy")
np.save(output_file, correlation_map)
print(f"Saved stacked noise ceiling to '{output_file}'")

valid_corrs = correlation_map[correlation_map != 0]
print(f"Mean noise ceiling correlation: {np.mean(valid_corrs):.4f}")
print(f"Std noise ceiling correlation: {np.std(valid_corrs):.4f}")
