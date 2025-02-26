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

os.chdir(r"C:\Users\adywi\OneDrive - unige.ch\Documents\Sarcasm_experiment\Irony_DeepLearning")
args = argparse.Namespace(
    img_size=[75, 92, 77],
    use_audio = True,
    use_text = False,
    use_context = False,
    use_base_features=True,
    use_pca=True, num_jobs = 15, alpha = 1, pca_threshold = 0.5)


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
    dataset_group.add_argument("--pca_threshold", type=float, nargs='+', default=[0.70, 0.80, 0.90],
                           help="List of explained variance thresholds for text PCA (default: [0.90, 0.95, 0.99]).")

    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--alpha", type=float, default=1.0,
                                help="Regularization strength for Ridge regression (default: 1.0).")
    analysis_group.add_argument("--num_jobs", type=int, default=-1,
                                help="Number of parallel jobs for voxel processing (default: -1 for all cores).")

    return parser.parse_args()


# Function to perform voxel-wise Ridge regression
def voxel_analysis(voxel, database_train, database_test, alpha):
    """Train Ridge regression and compute correlation for a given voxel."""
    df_train = database_train.get_voxel_values(voxel)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = df_train.drop(columns=["fmri_value"]).values
    y_train = df_train["fmri_value"].values  

    df_test = database_test.get_voxel_values(voxel)
    X_test = df_test.drop(columns=["fmri_value"]).values
    y_test = df_test["fmri_value"].values  

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    correlation = pearsonr(y_pred, y_test)[0] if np.std(y_test) > 0 else 0
    return voxel, correlation


# Main function
def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    print(f"Running with settings:\n"
          f"- Image size: {args.img_size}\n"
          f"- Mode: {args.mode}\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- n_component_text: {args.n_component_text}\n"
          f"- n_component_audio: {args.n_component_audio}\n"
          f"- Ridge alpha: {args.alpha}\n"
          f"- Number of parallel jobs: {args.num_jobs}")

    paths = analysis_helpers.get_paths()
    participant_list = os.listdir(paths["data_path"])
    train_participants, test_participants = train_test_split(participant_list, test_size=0.2, random_state=42)
    database_train = analysis_helpers.load_dataset(args, paths, train_participants)
    database_test = analysis_helpers.load_dataset(args, paths, test_participants)

    # Generate voxel list dynamically
    voxel_list = list(np.ndindex(tuple(args.img_size)))

    # Initialize correlation map
    correlation_map = np.zeros(tuple(args.img_size))

    # Parallel processing for all voxels
    results = Parallel(n_jobs=args.num_jobs)(
        delayed(voxel_analysis)(voxel, database_train, database_test, args.alpha) for voxel in voxel_list
    )

    # Store correlations and update the fMRI-sized array
    correlations = []  # List to store correlation values
    for voxel, corr in results:
        correlation_map[voxel] = corr
        correlations.append(corr)  # Collect correlations for mean calculation

    # Compute the mean correlation
    mean_correlation = np.mean(correlations)
    print(f"Mean correlation: {mean_correlation:.4f}")
    
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
    
    # Modify the result file name
    result_file = os.path.join(
        paths["results_path"],
        f"correlation_map_{args.mode}_{feature_str}.npy")

    np.save(result_file, correlation_map)

    end_time = time.time()  # Stop timing
    elapsed_time = end_time - start_time
    print(f"Correlation map saved as '{result_file}'")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

# Run the script
if __name__ == "__main__":
    main()
