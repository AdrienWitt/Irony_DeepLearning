#import scipy
import os
import numpy as np
import logging
from analysis_helpers import mult_diag, counter
import random
import itertools as itools
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
import sys

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

ridge_logger = logging.getLogger("ridge_corr")


os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
backend = 'loky'
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


def _compute_fold_corrs(fold_idx, train_idx, test_idx, stim, resp, valphas, normalpha, singcutoff, use_corr, logger):
    """Helper function to compute correlations for a single CV fold."""
    logger.info(f"Processing fold {fold_idx+1}/5...")
    Rstim, Pstim = stim[train_idx], stim[test_idx]
    Rresp, Presp = resp[train_idx], resp[test_idx]
    fold_corrs = ridge_corr_pred(
        Rstim=Rstim, Pstim=Pstim, Rresp=Rresp, Presp=Presp, valphas=valphas,
        normalpha=normalpha, singcutoff=singcutoff, use_corr=use_corr, logger=logger
    )
    return fold_corrs

def ridge_cv_participant(stim_df, resp, alphas, participant_ids, nboots=50,
                         corrmin=0.1, singcutoff=1e-10, normalpha=False, use_corr=True, return_wt=False,
                         normalize_stim=False, normalize_resp=True, n_jobs=1, with_replacement=False,
                         bootstrap_n_jobs=1, logger=ridge_logger):
    """Performs ridge regression with 5-fold participant-based CV after optimizing alpha using
    leave-one-participant-out (LOPO) bootstrapping.

    Parameters
    ----------
    stim_df : pandas.DataFrame, shape (T, N+P-1[+K-1])
        Stimuli with T time points, N features, P-1 participant covariates, K-1 task covariates.
    resp : array_like, shape (T, M)
        fMRI responses with T time points and M voxels.
    alphas : list or array_like, shape (A,)
        Ridge parameters to test (e.g., np.logspace(0, 3, 20)).
    participant_ids : array_like, shape (T,)
        Participant IDs for each time point.
    task_ids : array_like, shape (T,), optional
        Task IDs for task covariates.
    nboots : int, default 50
        Number of LOPO bootstrap iterations (e.g., 50 for 50 participants).
    corrmin : float, default 0.1
        Minimum correlation for logging.
    singcutoff : float, default 1e-10
        Singular value cutoff for SVD.
    normalpha : boolean, default False
        Normalize alphas by largest singular value.
    use_corr : boolean, default True
        Use correlation (True) or R-squared (False).
    return_wt : boolean, default True
        Return regression weights.
    normalize_stim : boolean, default False
        Z-score stimuli (False if pre-normalized).
    normalize_resp : boolean, default True
        Z-score responses (True for per-task-normalized data).
    n_jobs : int, default 1
        Number of parallel jobs for CV folds (-1 for all cores).
    with_replacement : boolean, default False
        Sample participants with replacement in LOPO bootstrap.
    bootstrap_n_jobs : int, default 1
        Number of parallel jobs for bootstrap iterations (-1 for all cores).

    Returns
    -------
    wt : array_like, shape (N+P-1[+K-1], M)
        Regression weights (empty if return_wt=False).
    corrs : array_like, shape (M,)
        Average correlation across 5 folds.
    valphas : array_like, shape (M,)
        Optimal alpha per voxel.
    fold_corrs : array_like, shape (M, 5)
        Correlations per voxel per fold.
    """
    stim = stim_df.to_numpy()
    if stim.shape[0] != resp.shape[0]:
        raise ValueError("stim_df and resp must have same number of time points.")

    stim = zs(stim) if normalize_stim else stim
    resp = zs(resp) if normalize_resp else resp

    if participant_ids is None:
        raise ValueError("participant_ids required for participant-based CV and bootstrapping.")

    logger.info("Optimizing alphas with LOPO bootstrapping...")
    valphas, _, _ = bootstrap_ridge_alpha_lopo(
        Rstim=stim, Rresp=resp, alphas=alphas, nboots=nboots, participant_ids=participant_ids,
        corrmin=corrmin, singcutoff=singcutoff, normalpha=normalpha, use_corr=use_corr,
        with_replacement=with_replacement, n_jobs=bootstrap_n_jobs, logger=logger
    )

    logger.info("Performing 5-fold participant-based CV...")
    gkf = GroupKFold(n_splits=5)
    fold_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=1)(
        delayed(_compute_fold_corrs)(
            fold_idx, train_idx, test_idx, stim, resp, valphas, normalpha, singcutoff, use_corr, logger
        )
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(stim, resp, groups=participant_ids))
    )

    fold_corrs = np.stack(fold_results, axis=1)
    corrs = np.mean(fold_corrs, axis=1)

    wt = []
    if return_wt:
        logger.info("Computing weights on full dataset...")
        wt = ridge(stim, resp, valphas, singcutoff=singcutoff, normalpha=normalpha, logger=logger)

    return wt, corrs, valphas, fold_corrs


def _bootstrap_iter(val_participant, Rstim, Rresp, alphas, participant_ids, corrmin, singcutoff, normalpha, use_corr, logger):
    """Helper function for one LOPO iteration."""
    heldinds = participant_ids == val_participant
    notheldinds = ~heldinds
    RRstim, PRstim = Rstim[notheldinds, :], Rstim[heldinds, :]
    RRresp, PRresp = Rresp[notheldinds, :], Rresp[heldinds, :]
    return ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas, corrmin=corrmin, singcutoff=singcutoff,
                      normalpha=normalpha, use_corr=use_corr, logger=logger)

def bootstrap_ridge_alpha_lopo(Rstim, Rresp, alphas, nboots, participant_ids,
                               corrmin=0.1, joined=None, singcutoff=1e-10, normalpha=False,
                               use_corr=True, with_replacement=False, n_jobs=1, logger=ridge_logger):
    """Performs leave-one-participant-out (LOPO) bootstrapping with parallelization to optimize ridge regression alphas.

    Parameters
    ----------
    n_jobs : int, default 1
        Number of parallel jobs for bootstrap iterations (-1 for all cores).
    [Other parameters unchanged]

    Returns
    -------
    valphas, allRcorrs, valinds
    """
    nresp, nvox = Rresp.shape
    if nresp != len(participant_ids):
        raise ValueError("participant_ids must have same length as Rresp rows.")
    valinds = []
    unique_participants = np.unique(participant_ids)
    n_participants = len(unique_participants)

    if nboots > n_participants and not with_replacement:
        raise ValueError(f"nboots ({nboots}) cannot exceed number of participants ({n_participants}) without replacement.")

    if with_replacement:
        participant_choices = np.random.choice(unique_participants, size=nboots, replace=True)
    else:
        participant_choices = list(unique_participants)
        np.random.shuffle(participant_choices)
        participant_choices = participant_choices[:nboots]

    logger.info(f"Running {nboots} LOPO bootstrap iterations with {n_jobs} jobs...")
    Rcmats = Parallel(n_jobs=n_jobs, backend=backend, verbose=1)(
        delayed(_bootstrap_iter)(
            val_participant, Rstim, Rresp, alphas, participant_ids, corrmin, singcutoff, normalpha, use_corr, logger
        )
        for val_participant in counter(participant_choices, countevery=1, total=nboots)
    )

    valinds = [participant_ids == val_participant for val_participant in participant_choices]

    if nboots > 0:
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None

    if nboots == 0:
        raise ValueError("At least one bootstrap iteration required.")

    logger.info("Finding best alpha for each voxel...")
    if joined is None:
        meanbootcorrs = allRcorrs.mean(2)
        bestalphainds = np.argmax(meanbootcorrs, 0)
        valphas = alphas[bestalphainds]
    else:
        valphas = np.zeros((nvox,))
        for jl in joined:
            jcorrs = allRcorrs[:, jl, :].mean(1).mean(1)
            bestalpha = np.argmax(jcorrs)
            valphas[jl] = alphas[bestalpha]

    return valphas, allRcorrs, valinds


def bootstrap_ridge_alpha(Rstim, Rresp, alphas, nboots, chunklen, nchunks, participant_ids,
                          corrmin=0.1, joined=None, singcutoff=1e-10, normalpha=False,
                          use_corr=True, logger=ridge_logger):
    """Performs participant-based bootstrapping to optimize ridge regression alphas without computing
    weights or final predictions. Returns only the optimal alphas and bootstrap correlations.

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features (including dummy-coded participants).
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M voxels.
    alphas : list or array_like, shape (A,)
        Ridge parameters to test.
    nboots : int
        Number of bootstrap samples.
    chunklen : int
        Approximate time points per participant for bootstrapping.
    nchunks : int
        Number of participants to hold out per bootstrap sample.
    participant_ids : array_like, shape (TR,)
        Participant IDs for each time point.
    corrmin : float, default 0.1
        Minimum correlation for logging.
    joined : None or list of array_like, default None
        Groups of voxels to share the same alpha.
    singcutoff : float, default 1e-10
        Singular value cutoff for SVD.
    normalpha : boolean, default False
        Whether to normalize alphas by the largest singular value.
    use_corr : boolean, default True
        Use correlation (True) or R-squared (False).

    Returns
    -------
    valphas : array_like, shape (M,)
        Optimal alpha per voxel.
    allRcorrs : array_like, shape (A, M, B)
        Correlations for each alpha, voxel, and bootstrap iteration.
    valinds : array_like, shape (TH, B)
        Validation indices for each bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    valinds = []
    unique_participants = np.unique(participant_ids)
    n_participants = len(unique_participants)

    if nchunks > n_participants:
        raise ValueError(f"nchunks ({nchunks}) cannot exceed number of participants ({n_participants}).")

    Rcmats = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.info(f"Bootstrap iteration {bi+1}/{nboots}...")
        # Randomly select participants for validation
        val_participants = random.sample(list(unique_participants), nchunks)
        heldinds = np.isin(participant_ids, val_participants)
        notheldinds = ~heldinds
        valinds.append(heldinds)

        RRstim = Rstim[notheldinds, :]
        PRstim = Rstim[heldinds, :]
        RRresp = Rresp[notheldinds, :]
        PRresp = Rresp[heldinds, :]

        # Evaluate correlations for each alpha
        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                           corrmin=corrmin, singcutoff=singcutoff,
                           normalpha=normalpha, use_corr=use_corr,
                           logger=logger)
        Rcmats.append(Rcmat)

    # Find best alphas
    if nboots > 0:
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None

    if nboots == 0:
        raise ValueError("At least one bootstrap iteration required for alpha optimization.")

    logger.info("Finding best alpha for each voxel...")
    if joined is None:
        meanbootcorrs = allRcorrs.mean(2)
        bestalphainds = np.argmax(meanbootcorrs, 0)
        valphas = alphas[bestalphainds]
    else:
        valphas = np.zeros((nvox,))
        for jl in joined:
            jcorrs = allRcorrs[:, jl, :].mean(1).mean(1)
            bestalpha = np.argmax(jcorrs)
            valphas[jl] = alphas[bestalpha]

    return valphas, allRcorrs, valinds

def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        U,S,Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(stim, full_matrices=False)

    UR = np.dot(U.T, np.nan_to_num(resp))
    
    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float,int)):
        alpha = np.ones(resp.shape[1]) * alpha
    
    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        #awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = Vh.T.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])
        wt[:,selvox] = awt

    return wt
    

def ridge_corr_pred(Rstim, Pstim, Rresp, Presp, valphas, normalpha=False,
                    singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. Returns the correlation 
    between predicted and actual [Presp], without ever computing the regression weights.
    This function assumes that each voxel is assigned a separate alpha in [valphas].

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    valphas : list or array_like, shape (M,)
        Ridge parameter for each voxel.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    corr : array_like, shape (M,)
        The correlation between each predicted response and each column of Presp.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = valphas * norm
    else:
        nalphas = valphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())

    ualphas = np.unique(nalphas)
    corr = np.zeros((Rresp.shape[1],))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        alpha_pred = PVh.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])

        if use_corr:
            corr[selvox] = (zPresp[:,selvox] * zs(alpha_pred)).mean(0)
        else:
            resvar = (Presp[:,selvox] - alpha_pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            corr[selvox] = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

    return corr


def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S / (S ** 2 + na ** 2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        pred = np.dot(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        logger.info(log_msg)
    
    return Rcorrs


def bootstrap_ridge(Rstim, Rresp, Pstim, Presp, alphas, nboots, chunklen, nchunks,
                    corrmin=0.1, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
                    use_corr=True, return_wt=True, logger=ridge_logger):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.
    
    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist 
    will be given the same regularization parameter (the one that is the best on average).
    
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M different responses. Each response should be Z-scored across
        time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times 
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this 
        product should be about 20 percent of the total length of the training data.
    corrmin : float in [0..1], default 0.2
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    joined : None or list of array_like indices, default None
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    singcutoff : float, default 1e-10
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean, default False
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    single_alpha : boolean, default False
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean, default True
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    return_wt : boolean, default True
        If True, this function will compute and return the regression weights after finding the best
        alpha parameter for each voxel. However, for very large models this can lead to memory issues.
        If false, this function will _not_ compute weights, but will still compute prediction performance
        on the prediction dataset (Pstim, Presp).
    
    Returns
    -------
    wt : array_like, shape (N, M)
        If [return_wt] is True, regression weights for N features and M responses. If [return_wt] is False, [].
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    valinds = [] # Will hold the indices into the validation data for each bootstrap
    
    Rcmats = []
    for bi in counter(range(nboots), countevery=1, total=nboots):
        logger.info("Selecting held-out test set..")
        allinds = range(nresp)
        indchunks = list(zip(*[iter(allinds)]*chunklen))
        random.shuffle(indchunks)
        heldinds = list(itools.chain(*indchunks[:nchunks]))
        notheldinds = list(set(allinds)-set(heldinds))
        valinds.append(heldinds)
        
        RRstim = Rstim[notheldinds,:]
        PRstim = Rstim[heldinds,:]
        RRresp = Rresp[notheldinds,:]
        PRresp = Rresp[heldinds,:]
        
        # Run ridge regression using this test set
        Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                           corrmin=corrmin, singcutoff=singcutoff,
                           normalpha=normalpha, use_corr=use_corr,
                           logger=logger)
        
        Rcmats.append(Rcmat)
    
    # Find best alphas
    if nboots>0:
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None
    
    if not single_alpha:
        if nboots==0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")
        
        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = allRcorrs[:,jl,:].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        if nboots==0:
            if len(alphas)==1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = allRcorrs.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]
        
        valphas = np.array([bestalpha]*nvox)
        logger.info("Best alpha = %0.3f"%bestalpha)

    if return_wt:
        # Find weights
        logger.info("Computing weights for each response using entire training set..")
        wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)

        # Predict responses on prediction set
        logger.info("Predicting responses for predictions set..")
        pred = np.dot(Pstim, wt)

        # Find prediction correlations
        nnpred = np.nan_to_num(pred)
        if use_corr:
            corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1]
                                            for ii in range(Presp.shape[1])]))
        else:
            resvar = (Presp-pred).var(0)
            Rsqs = 1 - (resvar / Presp.var(0))
            corrs = np.sqrt(np.abs(Rsqs)) * np.sign(Rsqs)

        return wt, corrs, valphas, allRcorrs, valinds
    else:
        # get correlations for prediction dataset directly
        corrs = ridge_corr_pred(Rstim, Pstim, Rresp, Presp, valphas, 
                                normalpha=normalpha, use_corr=use_corr,
                                logger=logger, singcutoff=singcutoff)

        return [], corrs, valphas, allRcorrs, valinds

def ridge_cv_participant_old(stim_df, resp, alphas, participant_ids, nboots=15, chunklen=10, nchunks=1,
                         corrmin=0.1, singcutoff=1e-10, normalpha=False, use_corr=True, return_wt=True,
                         normalize_stim=False, normalize_resp=True, n_jobs=1, logger=ridge_logger):
    """Performs ridge regression with 5-fold participant-based CV after optimizing alpha once using
    participant-based bootstrapping. Parallelizes CV folds using joblib.

    Parameters
    ----------
    stim_df : pandas.DataFrame, shape (T, N+P-1[+K-1])
        Stimuli with T time points, N features (e.g., text/audio embeddings), P-1 dummy-coded participant columns,
        and optionally K-1 dummy-coded task columns.
    resp : array_like, shape (T, M)
        fMRI responses with T time points and M voxels.
    alphas : list or array_like, shape (A,)
        Ridge parameters to test (e.g., np.logspace(0, 3, 20)).
    participant_ids : array_like, shape (T,)
        Participant IDs for each time point, required for CV and bootstrapping.
    nboots : int, default 15
        Number of bootstrap samples for alpha optimization.
    chunklen : int, default 10
        Approximate time points per participant for bootstrapping.
    nchunks : int, default 1
        Number of participants to hold out per bootstrap sample.
    corrmin : float, default 0.1
        Minimum correlation for logging.
    singcutoff : float, default 1e-10
        Singular value cutoff for SVD.
    normalpha : boolean, default False
        Whether to normalize alphas by the largest singular value.
    use_corr : boolean, default True
        Use correlation (True) or R-squared (False).
    return_wt : boolean, default True
        Return regression weights.
    normalize_stim : boolean, default False
        Whether to Z-score stimuli (set False if pre-normalized).
    normalize_resp : boolean, default True
        Whether to Z-score responses (set True for per-task-normalized data).
    n_jobs : int, default 1
        Number of parallel jobs for CV folds (-1 uses all available cores).

    Returns
    -------
    wt : array_like, shape (N+P-1[+K-1], M)
        Regression weights for N features, P-1 participant covariates, K-1 task covariates, and M voxels.
        Empty if return_wt=False.
    corrs : array_like, shape (M,)
        Average correlation across 5 folds per voxel.
    valphas : array_like, shape (M,)
        Optimal alpha per voxel.
    fold_corrs : array_like, shape (M, 5)
        Correlations per voxel per fold.
    """
    # Convert DataFrame to NumPy array
    stim = stim_df.to_numpy()
    if stim.shape[0] != resp.shape[0]:
        raise ValueError("stim_df and resp must have the same number of time points.")

    # Z-score stimuli and responses if requested
    stim = zs(stim) if normalize_stim else stim
    resp = zs(resp) if normalize_resp else resp

    # Validate participant_ids
    if participant_ids is None:
        raise ValueError("participant_ids must be provided for participant-based CV and bootstrapping.")

    # Optimize alpha on full dataset
    logger.info("Optimizing alphas with participant-based bootstrapping...")
    valphas, _, _ = bootstrap_ridge_alpha_lopo(
        Rstim=stim, Rresp=resp, alphas=alphas, nboots=nboots, chunklen=chunklen, nchunks=nchunks,
        participant_ids=participant_ids, corrmin=corrmin, singcutoff=singcutoff,
        normalpha=normalpha, use_corr=use_corr, logger=logger
    )

    # Perform 5-fold participant-based CV in parallel
    logger.info("Performing 5-fold participant-based CV...")
    gkf = GroupKFold(n_splits=5)
    fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_fold_corrs)(
            fold_idx, train_idx, test_idx, stim, resp, valphas, normalpha, singcutoff, use_corr, logger
        )
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(stim, resp, groups=participant_ids))
    )

    # Collect fold correlations
    fold_corrs = np.stack(fold_results, axis=1)  # Shape: (M, 5)

    # Average correlations across folds
    corrs = np.mean(fold_corrs, axis=1)

    # Compute weights on full dataset
    wt = []
    if return_wt:
        logger.info("Computing weights on full dataset...")
        wt = ridge(stim, resp, valphas, singcutoff=singcutoff, normalpha=normalpha, logger=logger)

    return wt, corrs, valphas, fold_corrs
