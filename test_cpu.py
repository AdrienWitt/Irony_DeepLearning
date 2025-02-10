import os
import time
import psutil
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import dataset  # Assuming this is a custom module
from sklearn.model_selection import train_test_split

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss  # returns RSS in bytes

# Function to get CPU usage
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)  # returns CPU usage as a percentage


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
    train_participants, test_participants = train_test_split(participant_list, test_size=0.2, random_state=42)

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
        mode=mode
    )

    return database_train, database_test

# Function to perform voxel-wise Ridge regression
def voxel_analysis(voxel, database_train, database_test):
    """Train Ridge regression and compute correlation for a given voxel."""
    df_train = database_train.get_voxel_values(voxel)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
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

# Main function
def main(database_train, database_test):
    start_time = time.time()  # Start timing

    # Initial resource usage
    initial_memory = get_memory_usage()
    initial_cpu = get_cpu_usage()
    print(f"Initial memory usage: {initial_memory / (1024 ** 2):.2f} MB")
    print(f"Initial CPU usage: {initial_cpu}%")

    print(f"Running with image size: {img_size}, mode: {mode}")


    voxel_list = list(np.ndindex(img_size))
    voxel = voxel_list[255000]

    # Before processing the voxel, track memory and CPU usage
    memory_before = get_memory_usage()
    cpu_before = get_cpu_usage()
    print(f"Memory usage before voxel processing: {memory_before / (1024 ** 2):.2f} MB")
    print(f"CPU usage before voxel processing: {cpu_before}%")

    result = voxel_analysis(voxel, database_train, database_test)

    # After processing the voxel, track memory and CPU usage again
    memory_after = get_memory_usage()
    cpu_after = get_cpu_usage()
    print(f"Memory usage after voxel processing: {memory_after / (1024 ** 2):.2f} MB")
    print(f"CPU usage after voxel processing: {cpu_after}%")

    # Print the correlation result for the voxel
    voxel, correlation = result
    print(f"Voxel {voxel} correlation: {correlation:.4f}")

    # Stop timing and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time for one voxel: {elapsed_time:.2f} seconds")

img_size = (75, 92, 77)
mode = 'text_audio'
paths = get_paths()
database_train, database_test = load_datasets(paths, img_size, mode)
main(database_train, database_test)