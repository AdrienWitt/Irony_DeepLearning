#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import pandas as pd
import logging
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import resample_to_img
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

import analysis_helpers
from ridge_cv import ridge_cv


args = argparse.Namespace(
    use_audio = False,
    use_text = False,
    use_base_features=True,
    use_text_weighted = True,
    use_audio_opensmile = True,
    include_tasks = ["irony", "sarcasm"],
    use_pca=True, num_jobs = 1, alpha = 0.1, pca_threshold = 0.5, use_umap = False, data_type = 'normalized_time', 
    random_seed = 42, n_perms = 1000)

# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Permutation test for multimodal fMRI encoding models."
    )
    parser.add_argument("--use_text", action="store_true")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_base_features", action="store_true", default=True)
    parser.add_argument("--use_text_weighted", action="store_true", default=True)
    parser.add_argument("--use_audio_opensmile", action="store_true", default=True)
    parser.add_argument("--use_pca", action="store_true", default=True)
    parser.add_argument("--use_umap", action="store_true")
    parser.add_argument("--pca_threshold", type=float, default=0.6)
    parser.add_argument("--include_tasks", type=str, nargs="+", default=["irony", "sarcasm"])
    parser.add_argument("--data_type", type=str, default="normalized_time")
    parser.add_argument("--n_splits", type=int, default=None,
                               help="Number of splits for cross-validation (default: number of participants for LOO CV).")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_jobs", type=int, default=4)
    parser.add_argument("--corrmin", type=float, default=0.0)
    parser.add_argument("--normalpha", action="store_true", default=True)
    parser.add_argument("--use_corr", action="store_true", default=True)
    parser.add_argument("--normalize_stim", action="store_true", default=False)
    parser.add_argument("--normalize_resp", action="store_true", default=True)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()


# ----------------------------------------------------------------------
# One permutation: permute modality blocks (with base features) independently
# ----------------------------------------------------------------------
def run_one_permutation(
    stim_df: pd.DataFrame,
    resp: np.ndarray,
    ids_list: np.ndarray,
    cols_text: list,
    cols_audio: list,
    cols_combined: list,
    valphas: np.ndarray,
    args: argparse.Namespace,
    seed: int,
):
    rng = np.random.RandomState(seed)
    stim_perm = stim_df.copy()

    # --- Within-participant independent shuffling per modality ---
    for pid in np.unique(ids_list):
        idx = np.where(ids_list == pid)[0]

        # TEXT block (embeddings + text base: context, semantic)
        perm_idx_text = rng.permutation(idx)
        stim_perm.loc[idx, cols_text] = stim_perm.loc[perm_idx_text, cols_text].values

        # AUDIO block (embeddings + prosody base)
        perm_idx_audio = rng.permutation(idx)
        stim_perm.loc[idx, cols_audio] = stim_perm.loc[perm_idx_audio, cols_audio].values

    # --- Ridge CV on permuted data (same as observed models) ---
    _, corr_text, _, _, _ = ridge_cv(
        stim_df=stim_perm[cols_text],       # includes base features!
        resp=resp,
        alphas=None,
        participant_ids=ids_list,
        nboots=0,
        n_splits=len(np.unique(ids_list)),
        corrmin=args.corrmin,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=False,
        normalize_stim=args.normalize_stim,
        normalize_resp=args.normalize_resp,
        n_jobs=args.num_jobs,
        with_replacement=False,
        optimize_alpha=False,
        valphas=valphas,
        logger=ridge_logger,
    )

    _, corr_audio, _, _, _ = ridge_cv(
        stim_df=stim_perm[cols_audio],      # includes base features!
        resp=resp,
        alphas=None,
        participant_ids=ids_list,
        nboots=0,
        n_splits=len(np.unique(ids_list)),
        corrmin=args.corrmin,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=False,
        normalize_stim=args.normalize_stim,
        normalize_resp=args.normalize_resp,
        n_jobs=args.num_jobs,
        with_replacement=False,
        optimize_alpha=False,
        valphas=valphas,
        logger=ridge_logger,
    )

    _, corr_comb, _, _, _ = ridge_cv(
        stim_df=stim_perm[cols_combined],   # includes ALL base + embeddings
        resp=resp,
        alphas=None,
        participant_ids=ids_list,
        nboots=0,
        n_splits=len(np.unique(ids_list)),
        corrmin=args.corrmin,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=False,
        normalize_stim=args.normalize_stim,
        normalize_resp=args.normalize_resp,
        n_jobs=args.num_jobs,
        with_replacement=False,
        optimize_alpha=False,
        valphas=valphas,
        logger=ridge_logger,
    )

    return corr_comb - np.maximum(corr_text, corr_audio)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    start_time = time.time()

    # --- Args & logging ---
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("perm_test")
    global ridge_logger
    ridge_logger = logging.getLogger("ridge_corr")
    logger.info("=== Starting permutation test (text_weighted + audio_opensmile + base) ===")

    # --- Load data ---
    paths = analysis_helpers.get_paths()
    participant_list = sorted(os.listdir(paths["data_path"]))

    icbm = datasets.fetch_icbm152_2009()
    mask = image.load_img(icbm["mask"])
    example_nii = nib.load(
        f"data/fmri/{args.data_type}/{participant_list[0]}/"
        f"{participant_list[0]}_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz"
    )
    resampled_mask = resample_to_img(mask, example_nii, interpolation="nearest")

    stim_df, resp, ids_list = analysis_helpers.load_dataset(
        args, paths, participant_list, resampled_mask
    )

    # --- Define feature blocks (including base features) ---
    # Text embeddings + PCA
    cols_text_emb = [c for c in stim_df.columns if c.startswith(("emb_weighted_", "pc_weighted_"))]

    # Text base: context, semantic (one-hot from categorical)
    cols_text_base = [c for c in stim_df.columns if c.startswith(("context_", "semantic_"))]

    # Audio embeddings + PCA
    cols_audio_emb = [c for c in stim_df.columns if c.startswith(("emb_audio_opensmile_", "pc_audio_opensmile_"))]

    # Audio base: prosody (one-hot)
    cols_audio_base = [c for c in stim_df.columns if c.startswith("prosody_")]

    # Final model inputs (same as observed analysis)
    cols_text = cols_text_emb + cols_text_base
    cols_audio = cols_audio_emb + cols_audio_base
    cols_combined = cols_text + cols_audio  # = all stimulus features

    logger.info(
        f"Text model: {len(cols_text)} features "
        f"(emb: {len(cols_text_emb)}, base: {len(cols_text_base)}) | "
        f"Audio model: {len(cols_audio)} features "
        f"(emb: {len(cols_audio_emb)}, base: {len(cols_audio_base)})"
    )

    # --- Load observed correlations and valphas ---
    r_comb_obs = np.load(
        "results_wholebrain_irosar/normalized_time/"
        "correlation_map_flat_audio_opensmile_text_weighted_base_5.npy"
    )
    r_text_obs = np.load(
        "results_wholebrain_irosar/normalized_time/"
        "correlation_map_flat_text_weighted_base_5.npy"
    )
    r_audio_obs = np.load(
        "results_wholebrain_irosar/normalized_time/"
        "correlation_map_flat_audio_opensmile_base_5.npy"
    )

    valphas_path = os.path.join(
        paths["results_path"][args.data_type],
        "valphas_audio_opensmile_text_weighted_base.npy",
    )
    valphas = np.load(valphas_path)

    delta_obs = r_comb_obs - np.maximum(r_text_obs, r_audio_obs)
    logger.info(f"Loaded observed Δr for {delta_obs.shape[0]} voxels.")

    # --- Run permutations ---
    rng = np.random.RandomState(args.random_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.n_perms)

    perm_results = Parallel(n_jobs=args.num_jobs)(
        delayed(run_one_permutation)(
            stim_df, resp, ids_list,
            cols_text, cols_audio, cols_combined,
            valphas, args, seed
        )
        for seed in seeds
    )
    delta_perm = np.vstack(perm_results)
    logger.info(f"Completed {args.n_perms} permutations.")

    # --- p-values + FDR ---
    pvals = (np.sum(delta_perm >= delta_obs[np.newaxis, :], axis=0) + 1) / (args.n_perms + 1)
    reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    logger.info(f"Significant voxels (FDR < 0.05): {np.sum(reject)}")

    # --- Save ---
    results_path = args.results_dir if args.results_dir else paths["results_path"][args.data_type]
    perm_dir = os.path.join(results_path, "permutation_results")
    os.makedirs(perm_dir, exist_ok=True)

    out_file = os.path.join(perm_dir, "perm_stats_textweighted_opensmile_with_base.np")
    np.save(
        out_file,
        delta_obs=delta_obs,
        delta_perm=delta_perm,
        pvals=pvals,
        pvals_fdr=pvals_fdr,
        reject=reject,
        cols_text=cols_text,
        cols_audio=cols_audio,
        cols_combined=cols_combined,
    )
    logger.info(f"Results saved to {out_file}")
    logger.info(f"Total time: {(time.time() - start_time)/60:.2f} min.")


if __name__ == "__main__":
    main()