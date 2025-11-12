import os
from sklearn.model_selection import train_test_split
import dataset
import pandas as pd
import numpy as np

def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": {
            "unsmoothed": os.path.join(base_path, "data", "fmri", "normalized_time_unsmoothed_flatten"),
            "normalized": os.path.join(base_path, "data", "fmri", "normalized"),
            "unormalized": os.path.join(base_path, "data", "fmri", "unormalized"),
            "normalized_time": os.path.join(base_path, "data", "fmri", "normalized_time_flatten")
        },
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "embeddings_audio_opensmile_path": os.path.join(base_path, "embeddings", "audio_opensmile"),
        "results_path": {
            "unsmoothed": os.path.join(base_path, "results_wholebrain_irosar", "unsmoothed"),
            "normalized": os.path.join(base_path, "results_wholebrain_irosar", "normalized"),
            "unormalized": os.path.join(base_path, "results_wholebrain_irosar", "unormalized"),
            "normalized_time": os.path.join(base_path, "results_wholebrain_irosar", "normalized_time")
        },
        "group_mask_path": os.path.join(base_path, "data", "fmri", "group_masks", "group_mask", "group_mask_threshold_0.85.nii.gz")
    }

    # Create all results directories
    for path in paths["results_path"].values():
        os.makedirs(path, exist_ok=True)
    
    return paths

def get_unique_filename(base_path, filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(base_path, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return os.path.join(base_path, new_filename)

def load_dataset(args, paths, participant_list, mask):
    """Loads the dataset using parsed arguments."""

    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"][args.data_type],  # Select the correct path based on data type
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

    #database = dataset.BaseDataset(participant_list=participant_list, **dataset_args)
    #data, data_fmri, ids_list = dataset.WholeBrainDataset(participant_list=participant_list, mask=mask, **dataset_args).create_data()
    data, data_fmri, ids_list = dataset.WholeBrainDataset(participant_list=participant_list, mask=mask, **dataset_args).create_data()

    
    return data, data_fmri, ids_list

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

def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx

import time
import logging

def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)
    
    for count, thing in enumerate(iterable):
        yield thing
        
        if not count%countevery:
            current_time = time.time()
            elapsed_time = max(current_time - start_time, 1e-8)
            rate = float(count + 1) / elapsed_time

            if rate>1: ## more than 1 item/second
                ratestr = "%0.2f items/second"%rate
            else: ## less than 1 item/second
                ratestr = "%0.2f seconds/item"%(rate**-1)
            
            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime('%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d"%(count+1, total)
            else:
                timestr = ""
                itemstr = "%d"%(count+1)

            formatted_str = "%s items complete (%s%s)"%(itemstr,ratestr,timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)
