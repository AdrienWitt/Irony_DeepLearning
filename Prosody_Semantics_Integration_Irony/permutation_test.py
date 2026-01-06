"""
Permutation testing utilities for voxel-wise correlation / encoding results.

This module provides a small, documented implementation of a permutation
procedure to compute voxel-wise p-values for correlations between predicted
and observed fMRI responses. It is intentionally small and dependency-light
so it can be shared on GitHub.

Functions
- `voxelwise_permutation_pvalues(resp, pred, n_permutations=1000, seed=None)`

Inputs should be numpy arrays of shape (n_samples, n_voxels). The function
returns an array of p-values (n_voxels,) representing the two-sided p-value
for the observed correlation against the permutation null.
"""

import numpy as np


def _corrs_per_voxel(a, b):
    """Compute Pearson correlation per column (voxel).

    a, b : arrays (n_samples, n_voxels)
    Returns: array (n_voxels,)
    """
    # center columns
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    denom = np.sqrt((a_c ** 2).sum(axis=0) * (b_c ** 2).sum(axis=0))
    # avoid div by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        r = ((a_c * b_c).sum(axis=0)) / (denom + 1e-12)
    # nan -> 0
    r = np.nan_to_num(r)
    return r


def voxelwise_permutation_pvalues(resp, pred, n_permutations=1000, seed=None):
    """Compute two-sided permutation p-values per voxel.

    Args:
        resp: np.ndarray (n_samples, n_voxels) observed responses
        pred: np.ndarray (n_samples, n_voxels) model predictions (same shape)
        n_permutations: int, number of permutations (default 1000)
        seed: optional random seed

    Returns:
        pvals: np.ndarray (n_voxels,) two-sided p-values
        r_obs: np.ndarray (n_voxels,) observed correlations
    """
    if resp.shape != pred.shape:
        raise ValueError('resp and pred must have same shape')

    rng = np.random.default_rng(seed)
    n_samples, n_voxels = resp.shape

    # observed correlations per voxel
    r_obs = _corrs_per_voxel(resp, pred)

    # allocate null distribution maxima/mins for two-sided test
    # We'll compute permutation correlations and count absolute exceedances
    exceed_counts = np.zeros(n_voxels, dtype=int)

    for i in range(n_permutations):
        # permute sample order of predictions (keeps temporal structure of resp)
        perm_idx = rng.permutation(n_samples)
        pred_perm = pred[perm_idx]
        r_perm = _corrs_per_voxel(resp, pred_perm)
        # two-sided: count where abs(r_perm) >= abs(r_obs)
        exceed_counts += (np.abs(r_perm) >= np.abs(r_obs)).astype(int)

    # p-value with +1 correction
    pvals = (exceed_counts + 1) / (n_permutations + 1)
    return pvals, r_obs


if __name__ == '__main__':
    # small smoke test
    import numpy as _np
    n, v = 50, 10
    _resp = _np.random.RandomState(0).randn(n, v)
    _pred = _resp * 0.2 + _np.random.RandomState(1).randn(n, v) * 0.8
    p, r = voxelwise_permutation_pvalues(_resp, _pred, n_permutations=100)
    print('pvals:', p)
    print('r_obs:', r)
