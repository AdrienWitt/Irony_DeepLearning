# -*- coding: utf-8 -*-

import numpy as np
from joblib import Parallel, delayed
import umap
from sklearn.manifold import trustworthiness
import os
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
import dataset  
from collections import Counter
from joblib import Parallel, delayed
import analysis_helpers

args = argparse.Namespace(
    img_size=[75, 92, 77],
    use_audio = True,
    use_text = True,
    use_context = False,
    use_base_features=True,
    use_pca=False, num_jobs = 15, alpha = 1, pca_threshold = 0.5)


# Function to compute trustworthiness score for different n_components
def optimize_umap_n_components(data, min_dim=2, max_dim=50, step=2):
    best_n_components = min_dim
    best_trustworthiness = 0.0

    for n_components in range(min_dim, max_dim + 1, step):
        reducer = umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(data)

        # Compute trustworthiness (1.0 = perfect, lower = worse)
        score = trustworthiness(data, reduced_data, n_neighbors=20)
        print(f"n_components={n_components}, Trustworthiness={score:.4f}")

        # Select the smallest n_components where trustworthiness stabilizes
        if score > best_trustworthiness:
            best_trustworthiness = score
            best_n_components = n_components

    return best_n_components

def explore_components(database, voxel):
    df_train = database.get_voxel_values(voxel)
    X_train = df_train.drop(columns=["fmri_value"])
    embedding_text = [col for col in X_train.columns if col.startswith("emb_text_")]
    embedding_audio = [col for col in X_train.columns if col.startswith("emb_audio_")]
    data_text = X_train[embedding_text]
    data_audio = X_train[embedding_audio]
    optimal_n_text = optimize_umap_n_components(data_text)
    optimal_n_audio = optimize_umap_n_components(data_audio)
    
    return optimal_n_text, optimal_n_audio

paths = analysis_helpers.get_paths()
participant_list = os.listdir(paths["data_path"])
database = analysis_helpers.load_dataset(args, paths, participant_list)

voxel_list = list(np.ndindex(tuple(args.img_size)))
top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
top_voxels = analysis_helpers.get_top_voxels(database, args.img_size, voxel_list, top_voxels_path)


for voxel in top_voxels[1:10]:
    results = explore_components(database, voxel)
     
best_comp_text = [result[0] for result in results]  # These are PCA components
best_comp_audio = [result[1] for result in results]    
    
most_common_comp_text = Counter(best_comp_text).most_common(1)[0][0]
most_common_comp_audio = Counter(best_comp_audio).most_common(1)[0][0]
print(f"Most common best PCA threshold across top voxels: {most_common_comp_text}")
print(f"Most common best alpha across top voxels: {most_common_comp_audio}")
