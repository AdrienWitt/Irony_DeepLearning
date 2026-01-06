"""
Ridge regression with GroupKFold cross-validation.
Focused, documented version used for encoding analysis.
"""

import numpy as np
import logging
from sklearn.model_selection import GroupKFold

logger = logging.getLogger('ridge_cv')


def zs(v):
    return (v - v.mean(0)) / (v.std(0) + 1e-10)


def ridge(stim, resp, alpha, normalpha=False):
    """Closed-form ridge using SVD.

    Returns weight matrix of shape (n_features+1, n_voxels) where the first
    row corresponds to the intercept.
    """
    X = np.column_stack([np.ones(stim.shape[0]), stim])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if normalpha:
        alpha = alpha * np.mean(S ** 2)
    S_ridge = S / (S ** 2 + alpha)
    W = Vt.T @ np.diag(S_ridge) @ U.T @ resp
    return W


def ridge_cv(stim_df, resp, alphas, participant_ids, n_splits=5,
             normalize_stim=False, normalize_resp=True, return_wt=False,
             normalpha=False, logger=None):
    if logger is None:
        logger = logging.getLogger('ridge_cv')

    X = stim_df.values
    if normalize_stim:
        X = zs(X)
    Y = resp
    if normalize_resp:
        Y = zs(Y)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(participant_ids))))
    splits = list(gkf.split(X, groups=participant_ids))

    mean_test_corrs = np.zeros((len(alphas), Y.shape[1]))
    mean_train_corrs = np.zeros((len(alphas), Y.shape[1]))
    best_weights = None
    best_alpha = alphas[0]
    best_overall = -np.inf

    logger.info(f"Starting CV with {len(splits)} folds")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]

        for ai, a in enumerate(alphas):
            W = ridge(Xtr, Ytr, a, normalpha=normalpha)
            pred_tr = Xtr @ W[1:] + W[0]
            pred_te = Xte @ W[1:] + W[0]

            # compute per-voxel correlations (safe fallback if constant prediction)
            tr_corrs = np.array([
                np.corrcoef(Ytr[:, v], pred_tr[:, v])[0, 1] if np.std(pred_tr[:, v])>0 else 0
                for v in range(Y.shape[1])
            ])
            te_corrs = np.array([
                np.corrcoef(Yte[:, v], pred_te[:, v])[0, 1] if np.std(pred_te[:, v])>0 else 0
                for v in range(Y.shape[1])
            ])

            mean_train_corrs[ai] += tr_corrs / len(splits)
            mean_test_corrs[ai] += te_corrs / len(splits)

            mean_te = te_corrs.mean()
            if mean_te > best_overall:
                best_overall = mean_te
                best_alpha = a
                if return_wt:
                    best_weights = W

    logger.info(f"Best alpha: {best_alpha} (mean test corr {best_overall:.4f})")
    return best_alpha, mean_test_corrs.mean(axis=0), best_weights, None, mean_train_corrs.mean(axis=0)
