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
#     use_pca=True, num_jobs = 15, alpha = 0.1, pca_threshold = 0.6)


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
    dataset_group.add_argument("--pca_threshold", type=float, nargs='+', default=0.60,
                           help="List of explained variance thresholds for text PCA (default: [0.90, 0.95, 0.99]).")

    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--alpha", type=float, default=1.0,
                                help="Regularization strength for Ridge regression (default: 1.0).")
    analysis_group.add_argument("--num_jobs", type=int, default=-1,
                                help="Number of parallel jobs for voxel processing (default: -1 for all cores).")

    return parser.parse_args()


def voxel_analysis(voxel, df_train, alpha):
    """Train Ridge regression and compute correlation for a given voxel using 5-fold CV."""
    # Create a unique random seed from voxel coordinates
    random_seed = int(voxel[0] * 10000 + voxel[1] * 100 + voxel[2])
    
    # Get voxel values and prepare data
    X_train = df_train.drop(columns=["fmri_value"]).values
    y_train = df_train["fmri_value"].values  

    # Perform 5-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Use a different random seed for the model
        model_seed = random_seed + fold
        ridge = Ridge(alpha=alpha, random_state=model_seed)
        ridge.fit(X_train_fold, y_train_fold)
        y_pred_fold = ridge.predict(X_val_fold)
        
        correlation = pearsonr(y_pred_fold, y_val_fold)[0] if np.std(y_val_fold) > 0 else 0
        cv_scores.append(correlation)
    
    # Use mean CV score
    mean_correlation = np.mean(cv_scores)
    print(f"{voxel} {mean_correlation}")
    return voxel, mean_correlation


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
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Ridge alpha: {args.alpha}\n"
          f"- Number of parallel jobs: {args.num_jobs}")

    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])[0:10]
    database_train = analysis_helpers.load_dataset(args, paths, participant_list)

    # Generate voxel list dynamically
    voxel_list = list(np.ndindex(tuple(args.img_size)))
    voxel_list = voxel_list[0:500]

    # Initialize correlation map
    correlation_map = np.zeros(tuple(args.img_size))

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

    # Store correlations and update the fMRI-sized array
    correlations = []  # List to store correlation values
    for voxel, corr in results:
        correlation_map[voxel] = corr
        correlations.append(corr)

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
    
    # Save correlation map
    result_file = os.path.join(
        paths["results_path"],
        f"correlation_map_{feature_str}.npy")
    np.save(result_file, correlation_map)
    print(f"\nCorrelation map saved as '{result_file}'")

    end_time = time.time()  # Stop timing
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

# Run the script
if __name__ == "__main__":
    main()
