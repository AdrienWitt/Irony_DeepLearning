# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:25:32 2025

@author: adywi
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import os
os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
import processing_helpers

root_folder = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\fMRI_study\data"
embeddings_text_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\text\statements"
embeddings_audio_path = r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning\embeddings\audio"

participant_list = os.listdir(root_folder)

embeddings_text_list = []
embeddings_audio_list = []

for participant in participant_list:
    participant_data_path = os.path.join(root_folder, participant)
    dfs = processing_helpers.load_dataframe(participant_data_path)
    for df in dfs.values():
        df = df.rename(columns=lambda x: x.strip())
        for index, row in df.iterrows():
            parts = row["Condition_name"].split("_")
            context_cond = parts[0]
            statement_cond = parts[1] 
            semantic = statement_cond[:2]
            situation = row["Situation"]
            embeddings_text = np.load(os.path.join(embeddings_text_path, f"{semantic}_{situation}_CLS.npy"))
            embeddings_text_list.append(embeddings_text)
            embeddings_audio = np.load(os.path.join(embeddings_audio_path, f"{row['Statement'].replace('.wav', '_layers5-6.npy')}"))
            embeddings_audio_list.append(embeddings_audio)

            
embeddings_text_df = pd.DataFrame(
    np.vstack(embeddings_text_list),
    columns=[f'embedding_text_{i}' for i in range(embeddings_text_list[0].shape[1])]
)            
                            
embeddings_audio_df = pd.DataFrame(
    np.vstack(embeddings_audio_list),
    columns=[f'embedding_audio_{i}' for i in range(embeddings_audio_list[0].shape[1])]
)

# Function to determine the number of components for a given variance threshold
def determine_components_for_variance(data, variance_threshold=0.95):
    """
    Determine the number of PCA components required to explain at least `variance_threshold` of the variance.
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Fit PCA without limiting the number of components
    pca = PCA()
    pca.fit(data_scaled)

    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components needed to exceed the variance threshold
    n_components = np.argmax(cumulative_explained_variance >= variance_threshold) + 1

    return n_components, cumulative_explained_variance, pca

# Determine the number of components for text embeddings
n_components_text, cumulative_variance_text, pca_text = determine_components_for_variance(embeddings_text_df)
print(f"Number of components for text embeddings (95% variance): {n_components_text}")

# Determine the number of components for audio embeddings
n_components_audio, cumulative_variance_audio, pca_audio = determine_components_for_variance(embeddings_audio_df)
print(f"Number of components for audio embeddings (95% variance): {n_components_audio}")

# Plot cumulative explained variance for text embeddings
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance_text, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(x=n_components_text, y=0.95, s=f'{n_components_text} components', color='red')
plt.title("Cumulative Explained Variance for Text Embeddings")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.show()

# Plot cumulative explained variance for audio embeddings
plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance_audio, marker='o', linestyle='--', color='g')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(x=n_components_audio, y=0.95, s=f'{n_components_audio} components', color='red')
plt.title("Cumulative Explained Variance for Audio Embeddings")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.show()