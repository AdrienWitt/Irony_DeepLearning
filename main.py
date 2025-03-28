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

# os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
# args = argparse.Namespace(
#     img_size=[75, 92, 77],
#     use_audio = True,
#     use_text = True,
#     use_base_features=True,
#     use_context = False,
#     use_pca=True, num_jobs = 1, alpha = 0.1, pca_threshold = 0.5, use_umap = False)

# voxel = (70, 70, 70)

# df_train = database_train.get_voxel_values(voxel)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # **Dataset-related arguments**
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--img_size", type=int, nargs=3, default=[75, 92, 77],
                                help="Size of fMRI images as three integers (default: 75 92 77).")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                                help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--use_text", action="store_true", 
                                help="Include text in dataset (default: False).")
    dataset_group.add_argument("--use_audio", action="store_true", 
                                help="Include audio in dataset (default: False).")
    dataset_group.add_argument("--use_context", action="store_true", 
                                help="Include context in dataset (default: False).")
    dataset_group.add_argument("--use_pca", action="store_true", 
                                help="Use PCA for embeddings with the a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--use_umap", action="store_true",
                                help="Use UMAP for dimensionality reduction (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.60,
                            help="Explained variance threshold for PCA (default: 0.60).")

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
        
    print(f"df_train shape: {df_train.shape}")
    # Extract features, target, and mask
    X = df_train.drop(columns=["fmri_value", "fmri_mask"]).values
    y = df_train["fmri_value"].values
    mask = df_train["fmri_mask"].values
    
    # Print data information for debugging
    print(f"Data shape for voxel {voxel}: X={X.shape}, y={y.shape}, mask={mask.shape}")
    print(f"First few rows of X: {X[:5]}")
    print(f"First few values of y: {y[:5]}")
    print(f"First few values of mask: {mask[:5]}")
    
    # Filter data based on mask BEFORE cross-validation
    X_filtered = X[mask == 1]
    y_filtered = y[mask == 1]
    
    # Check if there's any valid data after filtering
    if len(X_filtered) == 0:
        print(f"No valid data for voxel {voxel} after filtering")
        return voxel, [0] * 5, 0  # Return zeros for all folds and mean if no valid data
    
    # Perform 5-fold cross-validation on filtered data
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = []
    
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
    
    mean_correlation = np.mean(cv_scores)
    print(f"{voxel} {mean_correlation}")
    return voxel, cv_scores, mean_correlation


def process_voxel(args_tuple):
    # Unpack arguments
    voxel, df_train, alpha = args_tuple
    return voxel_analysis(voxel, df_train, alpha)


def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    print(f"Running with settings:\n"
          f"- Image size: {args.img_size}\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use context: {args.use_context}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- Use UMAP: {args.use_umap}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Ridge alpha: {args.alpha}\n"
          f"- Number of parallel jobs: {args.num_jobs}")

    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])
    #participant_list = os.listdir(paths["data_path"])[0:30]
    database_train = analysis_helpers.load_dataset(args, paths, participant_list)
        # Generate voxel list dynamically
    voxel_list = list(np.ndindex(tuple(args.img_size)))
    #voxel_list = voxel_list[40000:41000]

    # Initialize correlation maps (one for mean and one for individual folds)
    correlation_map_mean = np.zeros(tuple(args.img_size))
    correlation_map_folds = np.zeros(tuple(args.img_size) + (5,))  # Shape: (75, 92, 77, 5)

    # Method 2: Using multiprocessing Pool
    from multiprocessing import Pool

    # Prepare arguments for each voxel
    process_args = [
        (voxel, database_train.get_voxel_values(voxel), args.alpha)
        for voxel in voxel_list
    ]

    # Create a pool of workers and map the work
    with Pool(processes=args.num_jobs) as pool:
        results = pool.map(process_voxel, process_args)

    # Store correlations and update the fMRI-sized arrays
    correlations = []  # List to store mean correlation values
    for voxel, cv_scores, mean_corr in results:
        correlation_map_mean[voxel] = mean_corr
        correlation_map_folds[voxel] = cv_scores
        correlations.append(mean_corr)

    # Compute and print correlation statistics
    mean_correlation = np.mean(correlations)
    print(f"\nCorrelation Statistics:")
    print(f"Mean correlation: {mean_correlation:.4f}")
    print(f"Std correlation: {np.std(correlations):.4f}")
    print(f"Max correlation: {np.max(correlations):.4f}")
    print(f"Min correlation: {np.min(correlations):.4f}")
    
    features_used = []
    if args.use_text:
        features_used.append("text")
    if args.use_audio:
        features_used.append("audio")
    if args.use_context:
        features_used.append("context")
    if args.use_base_features:
        features_used.append("base")
    
    # Create a feature string (e.g., "text_audio" if both are enabled)
    feature_str = "_".join(features_used) if features_used else "nofeatures"
    
    # Save correlation maps
    result_file_mean = os.path.join(
        paths["results_path"],
        f"correlation_map_mean_{feature_str}.npy")
    result_file_folds = os.path.join(
        paths["results_path"],
        f"correlation_map_folds_{feature_str}.npy")
    
    np.save(result_file_mean, correlation_map_mean)
    np.save(result_file_folds, correlation_map_folds)
    print(f"\nCorrelation maps saved as:")
    print(f"- Mean correlations: '{result_file_mean}'")
    print(f"- Individual fold correlations: '{result_file_folds}'")

    end_time = time.time()  # Stop timing
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

# Run the script
if __name__ == "__main__":
    main()
