"""
Results analysis: compute voxel-wise p-values from saved model outputs.

This script provides a small helper function to load saved correlation maps and
predictions (or to re-run predictions if available) and compute permutation
p-values using `permutation_test.voxelwise_permutation_pvalues`.

It is intentionally small: it expects numpy files saved by `encoding.py` in the
`results_path` directory (see `analysis_helpers.get_paths()`). Typical files:
- `correlations.npy` (per-voxel correlations or correlation maps)
- `weights.npy` (optional weights)

The function `compute_and_save_pvalues` loads `resp` and `pred` (or constructs
predictions from `weights` if needed) and writes `pvalues.npy` and `r_obs.npy`.
"""

import os
import numpy as np
from analysis_helpers import get_paths
from permutation_test import voxelwise_permutation_pvalues


def compute_and_save_pvalues(results_dir, resp=None, pred=None, weights=None,
                             n_permutations=5000, seed=None):
    """Compute p-values and save outputs in `results_dir`.

    Args:
        results_dir (str): directory where to save `pvalues.npy` and `r_obs.npy`.
        resp (np.ndarray): observed responses (n_samples, n_voxels). If None,
            this function cannot compute p-values unless pred and resp are provided.
        pred (np.ndarray): model predictions (n_samples, n_voxels). If None and
            `weights` is provided, caller must provide stim matrix separately.
        weights (np.ndarray): optional; not used here but kept for API compatibility.
        n_permutations (int): number of permutations for p-values (default 5000).
        seed: RNG seed.

    Returns:
        pvals, r_obs arrays saved to disk.
    """
    os.makedirs(results_dir, exist_ok=True)

    if resp is None or pred is None:
        raise ValueError('resp and pred must be provided to compute p-values')

    pvals, r_obs = voxelwise_permutation_pvalues(resp, pred, n_permutations=n_permutations, seed=seed)

    np.save(os.path.join(results_dir, 'pvalues.npy'), pvals)
    np.save(os.path.join(results_dir, 'r_obs.npy'), r_obs)

    return pvals, r_obs


if __name__ == '__main__':
    # Example usage: user should edit paths accordingly
    paths = get_paths()
    results_dir = paths['results_path']

    # Look for example files (these are placeholders; adapt to your output names)
    corr_f = os.path.join(results_dir, 'correlations.npy')
    pred_f = os.path.join(results_dir, 'predictions.npy')
    resp_f = os.path.join(results_dir, 'resp.npy')

    if os.path.exists(resp_f) and os.path.exists(pred_f):
        resp = np.load(resp_f)
        pred = np.load(pred_f)
        compute_and_save_pvalues(results_dir, resp=resp, pred=pred, n_permutations=1000, seed=0)
    else:
        print('Please place `resp.npy` and `predictions.npy` in', results_dir)
