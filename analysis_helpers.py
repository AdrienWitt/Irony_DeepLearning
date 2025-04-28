import os
from sklearn.model_selection import train_test_split
import dataset
import pandas as pd
import numpy as np

def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": os.path.join(base_path, "data", "fmri", "group_masked_mc"),
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "embeddings_audio_opensmile_path": os.path.join(base_path, "embeddings", "audio_opensmile"),
        "results_path": os.path.join(base_path, "results"),
        "group_mask_path": os.path.join(base_path, "data", "fmri", "group_masks", "group_mask", "group_mask_threshold_0.85.nii.gz")
    }

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
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "embeddings_audio_opensmile_path": paths["embeddings_audio_opensmile_path"],
        "use_base_features": args.use_base_features,
        "use_text": args.use_text,
        "use_audio": args.use_audio,
        "use_audio_opensmile": args.use_audio_opensmile,
        "use_text_weighted": args.use_text_weighted,
        "pca_threshold": args.pca_threshold,
        "use_pca" : args.use_pca,
        "use_umap" : args.use_umap,
        "included_tasks": args.include_tasks
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
