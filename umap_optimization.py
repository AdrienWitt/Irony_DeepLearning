# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:33:51 2025

@author: adywi
"""
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



args = argparse.Namespace(
    img_size=[75, 92, 77],
    mode="text_audio",
    use_base_features=False,
    alpha=0.5,
    num_jobs=20,
    use_pca=False
)


def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": os.path.join(base_path, "data", "fmri"),
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text", "statements"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "results_path": os.path.join(base_path, "cv_results"),
    }

    # Create results directory if it doesn't exist
    os.makedirs(paths["results_path"], exist_ok=True)
    
    return paths

def load_dataset(args, paths):
    """Loads the dataset using parsed arguments."""
    participant_list = os.listdir(paths["data_path"])
    train_participants, test_participants = train_test_split(participant_list, test_size=0.2, random_state=42)

    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"],
        "img_size": tuple(args.img_size),
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "mode": args.mode,
        "use_base_features": args.use_base_features,
        "use_pca" : args.use_pca
    }

    database_train = dataset.BaseDataset(participant_list=train_participants, **dataset_args)

    return database_train

def get_top_voxels(database_train, img_size, voxel_list, top_voxels_path):
    if os.path.exists(top_voxels_path):
        df_voxels = pd.read_csv(top_voxels_path)
        top_voxels = [tuple(x) for x in df_voxels.to_records(index=False)]
        print(f"Loaded {len(top_voxels)} voxels from {top_voxels_path}")
    else:
        mean_activation = {voxel: np.mean(database_train.get_voxel_values(voxel)["fmri_value"].values) for voxel in voxel_list}
        threshold = np.percentile(list(mean_activation.values()), 90)
        top_voxels = [voxel for voxel, activation in mean_activation.items() if activation >= threshold]
        
        df_voxels = pd.DataFrame(top_voxels, columns=["X", "Y", "Z"])
        df_voxels.to_csv(top_voxels_path, index=False)
        print(f"Computed and saved {len(top_voxels)} top voxels to {top_voxels_path}")

    return top_voxels


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

def explore_components(database_train, voxel):
    df_train = database_train.get_voxel_values(voxel)
    X_train = df_train.drop(columns=["fmri_value"])
    embedding_text = [col for col in X_train.columns if col.startswith("emb_text_")]
    embedding_audio = [col for col in X_train.columns if col.startswith("emb_audio_")]
    data_text = X_train[embedding_text]
    data_audio = X_train[embedding_audio]
    optimal_n_text = optimize_umap_n_components(data_text)
    optimal_n_audio = optimize_umap_n_components(data_audio)
    
    return optimal_n_text, optimal_n_audio

paths = get_paths()
database_train = load_dataset(args, paths)

voxel_list = list(np.ndindex(tuple(args.img_size)))
top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
top_voxels = get_top_voxels(database_train, args.img_size, voxel_list, top_voxels_path)


for voxel in top_voxels[1:10]:
    results = explore_components(database_train, voxel)
     
best_comp_text = [result[0] for result in results]  # These are PCA components
best_comp_audio = [result[1] for result in results]    
    
most_common_comp_text = Counter(best_comp_text).most_common(1)[0][0]
most_common_comp_audio = Counter(best_comp_audio).most_common(1)[0][0]
print(f"Most common best PCA threshold across top voxels: {most_common_comp_text}")
print(f"Most common best alpha across top voxels: {most_common_comp_audio}")
