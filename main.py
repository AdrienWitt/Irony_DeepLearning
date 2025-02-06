import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Load dataset for training and testing
database_test = dataset.BaseDataset(
    participant_list=test_participants,
    data_path=data_path,
    fmri_data_path=fmri_data_path,
    img_size=img_size,
    embeddings_text_path=embeddings_text_path, embeddings_audio_path=embeddings_audio_path, mode = "audio_only"
)


# Compute mean activation per voxel in the training set
mean_activation = {}
for voxel in voxel_list:
    df_train = database_train.get_voxel_values(voxel)
    mean_activation[voxel] = np.mean(df_train["fmri_value"].values)

# Select the top 10 most activated voxels
top_voxels = sorted(mean_activation, key=mean_activation.get, reverse=True)[:10]

# Initialize correlation map
correlation_map = np.zeros(img_size)

def voxel_analysis(voxel):
    """Train Ridge regression and compute correlation for a given voxel."""
    df_train = database_train.get_voxel_values(voxel)
    X_train = df_train.drop(columns=["fmri_value"]).values
    y_train = df_train["fmri_value"].values  

    df_test = database_test.get_voxel_values(voxel)
    X_test = df_test.drop(columns=["fmri_value"]).values
    y_test = df_test["fmri_value"].values  

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    correlation = pearsonr(y_pred, y_test)[0] if np.std(y_test) > 0 else 0
    return voxel, correlation

# Parallel processing for all voxels
num_jobs = -1  # Use all CPU cores
results = Parallel(n_jobs=num_jobs)(delayed(voxel_analysis)(voxel) for voxel in voxel_list)

# Store correlations in the fMRI-sized array
for voxel, corr in results:
    correlation_map[voxel] = corr

print("Correlation map computed!")