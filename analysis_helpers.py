import os
from sklearn.model_selection import train_test_split
import dataset
import pandas as pd
import numpy as np

def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": os.path.join(base_path, "data", "fmri"),
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "results_path": os.path.join(base_path, "cv_results"),
    }

    # Create results directory if it doesn't exist
    os.makedirs(paths["results_path"], exist_ok=True)
    
    return paths

def get_unique_filename(base_path, filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(base_path, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return os.path.join(base_path, new_filename)

def load_dataset(args, paths, participant_list):
    """Loads the dataset using parsed arguments."""


    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"],
        "img_size": tuple(args.img_size),
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "use_base_features": args.use_base_features,
        "use_text": args.use_text,
        "use_audio": args.use_audio,
        "use_context": args.use_context,
        "pca_threshold": args.pca_threshold,
        "use_pca" : args.use_pca
    }

    database = dataset.BaseDataset(participant_list=participant_list, **dataset_args)
    
    return database

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
