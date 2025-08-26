import os
import time
import numpy as np
import analysis_helpers
import logging
from nilearn.image import resample_to_img
from nilearn import datasets, image
import nibabel as nib
import dataset
import argparse
import pandas as pd
from scipy.stats import pearsonr
from joblib import Parallel, delayed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ridge_corr")


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

zs = lambda v: (v-v.mean(0))/v.std(0)  # z-score function

def noise_ceiling_corr(raw_df, resp, participant_ids, match_columns, normalize_resp=True, n_jobs=1, logger=None):
  
    if raw_df.shape[0] != resp.shape[0]:
        raise ValueError("raw_df and resp must have same number of time points.")
    if not all(col in raw_df.columns for col in match_columns):
        raise ValueError("All match_columns must be present in raw_df.")
    if participant_ids.shape[0] != resp.shape[0]:
        raise ValueError("participant_ids must have same length as resp rows.")

    resp = zs(resp) if normalize_resp else resp

    logger = logger or logging.getLogger("noise_ceiling")
    logger.info("Computing noise ceiling with Pearson correlation across %d voxels...", resp.shape[1])

    # Check condition matches to ensure one time point per participant per condition
    def _check_condition_matches():
        logger.info("Checking condition matches across all data...")
        condition_groups = raw_df.groupby(match_columns).size().reset_index(name='count')
        for _, condition in condition_groups.iterrows():
            condition_mask = np.ones(len(raw_df), dtype=bool)
            for col in match_columns:
                condition_mask &= (raw_df[col] == condition[col]).values
            matching_indices = np.where(condition_mask)[0]
            matching_participants = participant_ids[matching_indices]
            participant_counts = pd.Series(matching_participants).value_counts()
            n_unique_participants = len(participant_counts)
            one_per_participant = (participant_counts == 1).all()
            if one_per_participant:
                logger.info(f"Condition {condition[match_columns].to_dict()}: "
                           f"{n_unique_participants}/{len(np.unique(participant_ids))} participants have exactly one time point.")
            else:
                logger.warning(f"Condition {condition[match_columns].to_dict()}: "
                              f"Some participants have multiple or zero time points: {participant_counts.to_dict()}")

    _check_condition_matches()

    def _compute_voxel_correlation(v):
        """Compute Pearson correlation for voxel v across all participants."""
        x_all, y_all = [], []
        for pid in np.unique(participant_ids):
            # Get indices for the current participant
            pid_idx = np.where(participant_ids == pid)[0]
            # Get indices for other participants
            other_idx = np.where(participant_ids != pid)[0]
            
            x_pid, y_pid = [], []
            for idx in pid_idx:
                # Find matching indices from other participants
                mask = np.ones(len(other_idx), dtype=bool)
                for col in match_columns:
                    mask &= (raw_df.iloc[other_idx][col] == raw_df.iloc[idx][col]).values
                matching_idx = other_idx[mask]
                if len(matching_idx) > 0:
                    # Compute mean response for voxel v from other participants
                    x_pid.append(np.nanmean(resp[matching_idx, v]))
                    y_pid.append(resp[idx, v])
                else:
                    x_pid.append(np.nan)
                    y_pid.append(resp[idx, v])
            
            x_all.extend(x_pid)
            y_all.extend(y_pid)
        
        # Convert to arrays and remove NaN pairs
        x_all = np.array(x_all)
        y_all = np.array(y_all)
        valid_mask = ~(np.isnan(x_all) | np.isnan(y_all))
        if np.sum(valid_mask) < 2:
            return np.nan
        corr, _ = pearsonr(x_all[valid_mask], y_all[valid_mask])
        return corr

    logger.info("Computing correlations for %d voxels using %d jobs...", resp.shape[1], n_jobs)
    noise_ceiling = Parallel(n_jobs=n_jobs)(
        delayed(_compute_voxel_correlation)(v) for v in range(resp.shape[1])
    )
    noise_ceiling = np.array(noise_ceiling)

    logger.info("Completed noise ceiling: mean correlation across voxels=%0.5f, max=%0.5f",
                np.nanmean(noise_ceiling), np.nanmax(noise_ceiling))

    return noise_ceiling


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
        use_pca=True, pca_threshold = 0.55, use_umap = False, data_type = 'normalized_time')
    
    stim_df, resp, ids_list, raw_df = load_dataset(args, paths, participant_list, resampled_mask)
    
    # Compute noise ceiling
    noise_ceiling = noise_ceiling_corr(
        raw_df=raw_df,
        resp=resp,
        participant_ids=ids_list,
        match_columns=match_columns,
        normalize_resp=True,
        logger=logger,
        n_jobs = 50
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "noise_ceiling_corr.npy"), noise_ceiling)
    logger.info(f"Saved noise ceiling correlations to {args.output_dir}")

    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    
   #  # Handle precomputed valphas
   #  valphas_path = os.path.join( "results_wholebrain_irosar/normalized_time", "valphas_audio_opensmile_text_weighted_base.npy")
   #  valphas = np.load(valphas_path)
    
   #  # Compute noise ceiling
   #  mean_noise_ceiling, fold_noise_ceilings = noise_ceiling(
   #      raw_df=raw_df,
   #      resp=resp,
   #      participant_ids=ids_list,
   #      match_columns=match_columns,
   #      valphas=valphas,
   #      n_splits=5,
   #      normalize_resp=True,
   #      n_jobs=args.num_jobs,
   #      logger=logger
   #  )
    
   # # Save results
   #  output_dir = os.path.join("results_wholebrain_irosar", args.data_type)
   #  os.makedirs(output_dir, exist_ok=True)
   #  np.save(os.path.join(output_dir, "mean_noise_ceiling.npy"), mean_noise_ceiling)
   #  np.save(os.path.join(output_dir, "fold_noise_ceilings.npy"), fold_noise_ceilings)
   #  logger.info(f"Saved noise ceiling results to {output_dir}")

   #  end_time = time.time()
   #  logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
 
   #  logger.info(f"Loaded data: features_df {raw_df.shape}, resp {resp.shape}, raw_df {raw_df.shape}")

if __name__ == "__main__":
    main()