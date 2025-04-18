import os
import time
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import dataset  
import analysis_helpers
from sklearn.metrics import r2_score


# os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
# args = argparse.Namespace(
#     use_audio = True,
#     use_text = False,
#     use_base_features=True,
#     use_text_weighted = True,
#     use_pca=True, num_jobs = 1, alpha = 0.1, pca_threshold = 0.5, use_umap = False)


# df_train = database_train.get_voxel_values(((50, 50, 50)))

# Set a reliable temporary directory for joblib
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # **Dataset-related arguments**
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                                help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--use_text", action="store_true", 
                                help="Include text in dataset (default: False).")
    dataset_group.add_argument("--use_audio", action="store_true", 
                                help="Include audio in dataset (default: False).")
    dataset_group.add_argument("--use_text_weighted", action="store_true", 
                                help="Include text_weighted in dataset (default: False).")
    dataset_group.add_argument("--use_pca", action="store_true", 
                                help="Use PCA for embeddings with the a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--use_umap", action="store_true",
                                help="Use UMAP for dimensionality reduction (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.60,
                            help="Explained variance threshold for PCA (default: 0.60).")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony", "prosody", "semantic", "tom"],
                            help="List of tasks to include (default: all available tasks).")
    

    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--alpha", type=float, default=1.0,
                                help="Regularization strength for Ridge regression (default: 1.0).")
    analysis_group.add_argument("--num_jobs", type=int, default=-1,
                                help="Number of parallel jobs for voxel processing (default: -1 for all cores).")

    return parser.parse_args()

def voxel_analysis(voxel, df_train, alpha):
    """Train Ridge regression and compute correlation for a given voxel using 5-fold CV."""
    random_seed = int(voxel[0] * 10000 + voxel[1] * 100 + voxel[2])
            # Extract features, target, and mask
    X = df_train.drop(columns=["fmri_value"]).values
    y = df_train["fmri_value"].values
        
    valid_idx = y != 0 ## exclude background values
    X_filtered = X[valid_idx]
    y_filtered = y[valid_idx]

    print("X_filtered length: ", len(X_filtered))
    
    # Exclude background
    if len(X_filtered) < len(X)*0.5:
        print(f"No enough data for voxel {voxel} after filtering")
        return voxel, [0] * 5, 0, [0] * 5, 0  # Return zeros for correlations and R^2
    
    # Perform 5-fold cross-validation on filtered data
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_filtered)):
        # Split filtered data
        X_train_fold = X_filtered[train_idx]
        y_train_fold = y_filtered[train_idx]
        X_val_fold = X_filtered[val_idx]
        y_val_fold = y_filtered[val_idx]
        
        # Train model
        model_seed = random_seed + fold
        ridge = Ridge(alpha=alpha, random_state=model_seed)
        ridge.fit(X_train_fold, y_train_fold)
        
        # Predict and compute correlation
        y_pred_fold = ridge.predict(X_val_fold)
        
        if np.std(y_val_fold) > 0:
            correlation = pearsonr(y_pred_fold, y_val_fold)[0]
        else:
            correlation = 0
            print(f"Standard deviation is 0 in fold {fold} for voxel {voxel}")
        
        cv_scores.append(correlation)
        
        r2 = r2_score(y_val_fold, y_pred_fold)
        r2_scores.append(r2)
    
    mean_correlation = np.mean(cv_scores)
    mean_r2 = np.mean(r2_scores)
    print(f"{voxel} corr: {mean_correlation:.4f}, R^2: {mean_r2:.4f}")
    return voxel, cv_scores, mean_correlation, r2_scores, mean_r2


def process_voxel(voxel, df_train, alpha):
    return voxel_analysis(voxel, df_train, alpha)

def adjust_alpha(database_train, args):
    df = database_train.data.drop(columns=["fmri_value"])
    if args.use_audio and args.use_text_weighted:
        alpha = args.alpha
    else: 
        alpha = args.alpha * df.shape[1] / 84 ## 79 is the total number of features with PCA 0.50 threshold for text and audio
        print(f"- Use correced alpha: {alpha}")
    return alpha
    

def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    print(f"Running with settings:\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use text_weighted: {args.use_text_weighted}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- Use UMAP: {args.use_umap}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Ridge alpha: {args.alpha}\n"
          f"- Number of parallel jobs: {args.num_jobs}\n"
          f"- Included tasks: {', '.join(args.include_tasks)}")

    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])
    participant_list = os.listdir(paths["data_path"])[0:10]
    database_train = analysis_helpers.load_dataset(args, paths, participant_list)
    
    alpha = adjust_alpha(database_train, args)
    
    # Generate voxel list dynamically
    img_size = (79, 95, 79)
    voxel_list = list(np.ndindex(img_size))
    # voxel_list = voxel_list[40000:50000]

    # Initialize correlation and R^2 maps
    correlation_map_mean = np.zeros(img_size)
    correlation_map_folds = np.zeros(img_size + (5,))
    r2_map_mean = np.zeros(img_size)
    r2_map_folds = np.zeros(img_size + (5,))


    print(f"Processing {len(voxel_list)} voxels")

    # Configure parallel processing
    n_jobs = args.num_jobs if args.num_jobs > 0 else os.cpu_count()
    backend = 'loky'  # Use loky backend which is more robust

    try:
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=1)(
            delayed(process_voxel)(voxel, database_train.get_voxel_values(voxel), alpha) 
            for voxel in voxel_list
        )
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        print("Falling back to sequential processing...")
        # Fallback to sequential processing
        results = []
        for voxel in voxel_list:
            try:
                result = process_voxel(voxel, database_train.get_voxel_values(voxel), args.alpha)
                results.append(result)
            except Exception as e:
                print(f"Error processing voxel {voxel}: {str(e)}")
                continue

    # Store correlations and update the fMRI-sized arrays
    correlations = []
    r2_values = []
    for voxel, cv_scores, mean_corr, r2_scores, mean_r2 in results:
        correlation_map_mean[voxel] = mean_corr
        correlation_map_folds[voxel] = cv_scores
        r2_map_mean[voxel] = mean_r2
        r2_map_folds[voxel] = r2_scores
        correlations.append(mean_corr)
        r2_values.append(mean_r2)

    # Compute and print correlation statistics
    mean_correlation = np.mean(correlations)
    mean_r2 = np.mean(r2_values)
    print(f"\nStatistics:")
    print(f"Mean correlation: {mean_correlation:.4f}, Std: {np.std(correlations):.4f}")
    print(f"Mean R^2: {mean_r2:.4f}, Std: {np.std(r2_values):.4f}")
    print(f"Max R^2: {np.max(r2_values):.4f}, Min R^2: {np.min(r2_values):.4f}")
    
    features_used = []
    if args.use_text:
        features_used.append("text")
    if args.use_audio:
        features_used.append("audio")
    if args.use_text_weighted:
        features_used.append("text_weighted")
    if args.use_base_features:
        features_used.append("base")
    # Create a feature string (e.g., "text_audio" if both are enabled)
    feature_str = "_".join(features_used) if features_used else "nofeatures"
    
    # Create task code string (first 3 letters of each task)
    task_code = "_".join([task[:3] for task in args.include_tasks])
    
    # Save maps
    result_file_mean = os.path.join(paths["results_path"], f"correlation_map_mean_{feature_str}_{task_code}.npy")
    result_file_folds = os.path.join(paths["results_path"], f"correlation_map_folds_{feature_str}_{task_code}.npy")
    r2_file_mean = os.path.join(paths["results_path"], f"r2_map_mean_{feature_str}_{task_code}.npy")
    r2_file_folds = os.path.join(paths["results_path"], f"r2_map_folds_{feature_str}_{task_code}.npy")
    
    np.save(result_file_mean, correlation_map_mean)
    np.save(result_file_folds, correlation_map_folds)
    np.save(r2_file_mean, r2_map_mean)
    np.save(r2_file_folds, r2_map_folds)
    print(f"\nMaps saved as:")
    print(f"- Mean correlations: {result_file_mean}")
    print(f"- Fold correlations: {result_file_folds}")
    print(f"- Mean R^2: {r2_file_mean}")
    print(f"- Fold R^2: {r2_file_folds}")
    
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time):.2f} seconds")

# Run the script
if __name__ == "__main__":
    main()
