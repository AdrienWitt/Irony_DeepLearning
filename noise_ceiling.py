import os
import time
import numpy as np
import analysis_helpers
import logging
from nilearn.image import resample_to_img
from nilearn import datasets, image
import nibabel as nib
import dataset
import pandas as pd
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from ridge_cv import ridge_corr_pred

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
        "fmri_data_path": paths["fmri_data_path"][args.data_type],
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

    dataset_obj = dataset.WholeBrainDatasetWithRaw(
        participant_list=participant_list,
        mask=mask,
        **dataset_args
    )

    return dataset_obj.data, dataset_obj.fmri_data, dataset_obj.ids_list, dataset_obj.raw_df

zs = lambda v: (v-v.mean(0))/v.std(0)  # z-score function

def noise_ceiling(raw_df, resp, participant_ids, match_columns, valphas, n_splits=5, normalize_resp=True, n_jobs=1, logger=None):
    """
    Computes the noise ceiling using GroupKFold cross-validation with ridge regression.
    For each voxel, uses the average response of all other participants (from the entire dataset) under
    matching conditions as the predictor to train a ridge regression model, then predicts the actual
    response of the current participant in the test set and correlates with actual responses.
    """
    if raw_df.shape[0] != resp.shape[0]:
        raise ValueError("raw_df and resp must have same number of time points.")
    if not all(col in raw_df.columns for col in match_columns):
        raise ValueError("All match_columns must be present in raw_df.")
    if participant_ids.shape[0] != resp.shape[0]:
        raise ValueError("participant_ids must have same length as resp rows.")
    if valphas.shape[0] != resp.shape[1]:
        raise ValueError("valphas must have same length as number of voxels.")

    resp = zs(resp) if normalize_resp else resp

    unique_participants = np.unique(participant_ids)
    n_participants = len(unique_participants)
    n_splits = min(n_splits, n_participants)

    logger = logger or logging.getLogger("noise_ceiling")
    logger.info("Computing noise ceiling with %d-fold GroupKFold CV across %d voxels using ridge regression...", n_splits, resp.shape[1])

    def _compute_fold_noise_ceiling(fold_idx, train_idx, test_idx):
        logger.info(f"Processing noise ceiling fold {fold_idx+1}/{n_splits}...")
        corrs = np.zeros(resp.shape[1])
        train_X = np.zeros((len(train_idx), resp.shape[1]))
        test_X = resp[test_idx]

        for i, idx_i in enumerate(train_idx):
            mask = np.ones(len(raw_df), dtype=bool)
            for col in match_columns:
                mask &= (raw_df[col] == raw_df.iloc[idx_i][col]).values
            mask &= (participant_ids != participant_ids[idx_i])
            matching_idx = np.where(mask)[0]
            train_X[i] = np.nanmean(resp[matching_idx], axis=0) if len(matching_idx) > 0 else np.full(resp.shape[1], np.nan)

        corrs = ridge_corr_pred(train_X, test_X, resp[train_idx], resp[test_idx], valphas,
                                normalpha=False, singcutoff=1e-10, use_corr=True, logger=logger)

        return corrs

    gkf = GroupKFold(n_splits=n_splits)
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_fold_noise_ceiling)(fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(resp, resp, groups=participant_ids))
    )

    fold_noise_ceilings = np.stack(fold_results, axis=1) if fold_results else np.array([])
    mean_noise_ceiling = np.nanmean(fold_noise_ceilings, axis=1) if fold_noise_ceilings.size > 0 else np.array([])

    logger.info("Completed noise ceiling: mean correlation across voxels=%0.5f, max=%0.5f",
                np.nanmean(mean_noise_ceiling), np.nanmax(mean_noise_ceiling))

    return mean_noise_ceiling, fold_noise_ceilings

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

    # Define arguments directly
    class Args:
        use_audio = False
        use_text = False
        use_base_features = True
        use_text_weighted = True
        use_audio_opensmile = True
        include_tasks = ["irony", "sarcasm"]
        use_pca = True
        pca_threshold = 0.55
        use_umap = False
        data_type = 'normalized_time'
        output_dir = "results_wholebrain_irosar/normalized_time"
        valphas_path = "results_wholebrain_irosar/normalized_time/valphas_audio_opensmile_text_weighted_base.npy"

    args = Args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    stim_df, resp, ids_list, raw_df = load_dataset(args, paths, participant_list, resampled_mask)
    logger.info(f"Dataset loaded: {resp.shape[0]} time points, {resp.shape[1]} voxels, {len(np.unique(ids_list))} participants")

    # Load precomputed valphas
    logger.info(f"Loading valphas from {args.valphas_path}...")
    valphas = np.load(args.valphas_path)
    logger.info(f"Loaded valphas with shape {valphas.shape}")

    # Compute noise ceiling
    logger.info("Starting noise ceiling computation...")
    mean_noise_ceiling, fold_noise_ceilings = noise_ceiling(
        raw_df=raw_df,
        resp=resp,
        participant_ids=ids_list,
        match_columns=match_columns,
        valphas=valphas,
        n_splits=5,
        normalize_resp=False,
        n_jobs=50,
        logger=logger
    )

    # Save results
    mean_output_path = os.path.join(args.output_dir, "mean_noise_ceiling.npy")
    fold_output_path = os.path.join(args.output_dir, "fold_noise_ceilings.npy")
    logger.info(f"Saving mean noise ceiling to {mean_output_path}...")
    np.save(mean_output_path, mean_noise_ceiling)
    logger.info(f"Saving fold noise ceilings to {fold_output_path}...")
    np.save(fold_output_path, fold_noise_ceilings)

    # Log completion time
    elapsed_time = time.time() - start_time
    logger.info(f"Completed in {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main()
