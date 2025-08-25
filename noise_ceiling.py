import os
import time
import numpy as np
import analysis_helpers
import logging
from ridge_cv import noise_ceiling_check
from nilearn.image import resample_to_img
from nilearn import datasets, image
import nibabel as nib
import dataset
import argparse
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
ridge_logger = logging.getLogger("ridge_corr")


def load_dataset(args, paths, participant_list, mask):
    """Loads the dataset using parsed arguments."""

    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"][args.data_type],  # pick based on type
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "embeddings_audio_opensmile_path": paths["embeddings_audio_opensmile_path"],
        "use_base_features": args.use_base_features,
        "use_text": args.use_text,
        "use_audio": args.use_audio,
        "use_audio_opensmile": args.use_audio_opensmile,
        "use_text_weighted": args.use_text_weighted,
        "pca_threshold": args.pca_threshold,
        "use_pca": args.use_pca,
        "use_umap": args.use_umap,
        "included_tasks": args.include_tasks,
    }

    # Instantiate dataset (this will run __init__ + create_data automatically)
    dataset_obj = dataset.WholeBrainDatasetWithRaw(
        participant_list=participant_list,
        mask=mask,
        **dataset_args
    )

    # Return everything
    return dataset_obj.data, dataset_obj.fmri_data, dataset_obj.ids_list, dataset_obj.raw_df


def main():
    start_time = time.time()

    # Load paths and participants
    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])

    # Load mask
    icbm = datasets.fetch_icbm152_2009()
    mask_path = icbm['mask']
    mask = image.load_img(mask_path)
    exemple_data = nib.load("data/fmri/normalized_time/p01/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')

    match_columns = ['context', 'semantic', 'prosody', 'task', 'situation']    

    args = argparse.Namespace(
        use_audio = False,
        use_text = False,
        use_base_features=True,
        use_text_weighted = True,
        use_audio_opensmile = True,
        include_tasks = ["irony", "sarcasm"],
        use_pca=True, num_jobs = 50, pca_threshold = 0.55, use_umap = False, data_type = 'normalized_time')
    
    stim_df, resp, ids_list, raw_df = load_dataset(args, paths, participant_list, resampled_mask)
    
    # Handle precomputed valphas
    valphas_path = os.path.join( "results_wholebrain_irosar/normalized_time", "valphas_audio_opensmile_text_weighted_base.npy")
    valphas = np.load(valphas_path)
    
    # Compute noise ceiling
    mean_noise_ceiling, fold_noise_ceilings = noise_ceiling_check(
        raw_df=raw_df,
        resp=resp,
        participant_ids=ids_list,
        match_columns=match_columns,
        valphas=valphas,
        n_splits=5,
        normalize_resp=True,
        n_jobs=args.num_jobs,
        logger=ridge_logger
    )
    
   # Save results
    output_dir = os.path.join("results_wholebrain_irosar", args.data_type)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mean_noise_ceiling.npy"), mean_noise_ceiling)
    np.save(os.path.join(output_dir, "fold_noise_ceilings.npy"), fold_noise_ceilings)
    ridge_logger.info(f"Saved noise ceiling results to {output_dir}")

    end_time = time.time()
    ridge_logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
 
    ridge_logger.info(f"Loaded data: features_df {raw_df.shape}, resp {resp.shape}, raw_df {raw_df.shape}")

if __name__ == "__main__":
    main()