import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import Parallel, delayed
import dataset  # Assuming this is a custom module


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # **Dataset-related arguments**
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--img_size", type=int, nargs=3, default=[75, 92, 77],
                               help="Size of fMRI images as three integers (default: 75 92 77).")
    dataset_group.add_argument("--mode", type=str, choices=["base_features", "audio", "text", "text_audio"], 
                               default="base_features", help="Mode for dataset loading (default: base_features).")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                               help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--n_component_text", type=int, default=22, 
                               help="Number of PCA components for text embeddings (default: 22).")
    dataset_group.add_argument("--n_component_audio", type=int, default=217, 
                               help="Number of PCA components for audio embeddings (default: 217).")

    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--alpha_values", type=float, nargs='+', default=[1, 5, 10, 15, 20, 30, 50],
                        help="List of alpha values for Ridge regression (default: [1, 5, 10, 15, 20, 30, 50]).")
    analysis_group.add_argument("--num_jobs", type=int, default=20,
                                help="Number of parallel jobs for voxel processing (default: -1 for all cores).")

    return parser.parse_args()

# Function to set up paths dynamically
def get_paths():
    base_path = os.getcwd()  # Gets the working directory where the script is executed

    paths = {
        "data_path": os.path.join(base_path, "data", "behavioral"),
        "fmri_data_path": os.path.join(base_path, "data", "fmri"),
        "embeddings_text_path": os.path.join(base_path, "embeddings", "text", "statements"),
        "embeddings_audio_path": os.path.join(base_path, "embeddings", "audio"),
        "results_path": os.path.join(base_path, "cv_results"),
    }

    # Create results directory if it doesn't exist
    os.makedirs(paths["results_path"], exist_ok=True)
    
    return paths

# Function to ensure a unique filename by appending a number if needed
def get_unique_filename(base_path, filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(base_path, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return os.path.join(base_path, new_filename)

# Function to load dataset and split participants
def load_dataset(args, paths):
    """Loads the dataset using parsed arguments."""
    participant_list = os.listdir(paths["data_path"])
    train_participants, test_participants = train_test_split(participant_list, test_size=0.2, random_state=42)

    dataset_args = {
        "data_path": paths["data_path"],
        "fmri_data_path": paths["fmri_data_path"],
        "img_size": tuple(args.img_size),
        "embeddings_text_path": paths["embeddings_text_path"],
        "embeddings_audio_path": paths["embeddings_audio_path"],
        "mode": args.mode,
        "use_base_features": args.use_base_features,
        "n_component_text": args.n_component_text,
        "n_component_audio": args.n_component_audio
    }

    database_train = dataset.BaseDataset(participant_list=train_participants, **dataset_args)

    return database_train

# Function to get top 10% most activated voxels
def get_top_voxels(database_train, img_size, voxel_list, top_voxels_path):
    if os.path.exists(top_voxels_path):
        df_voxels = pd.read_csv(top_voxels_path)
        top_voxels = [tuple(x) for x in df_voxels.to_records(index=False)]
        print(f"Loaded {len(top_voxels)} voxels from {top_voxels_path}")
    else:
        mean_activation = {voxel: np.mean(database_train.get_voxel_values(voxel)["fmri_value"].values) for voxel in voxel_list}
        threshold = np.percentile(list(mean_activation.values()), 90)
        top_voxels = [voxel for voxel, activation in mean_activation.items() if activation >= threshold]
        
        df_voxels = pd.DataFrame(top_voxels, columns=["X", "Y", "Z"])
        df_voxels.to_csv(top_voxels_path, index=False)
        print(f"Computed and saved {len(top_voxels)} top voxels to {top_voxels_path}")

    return top_voxels

# Function to perform voxel-wise Ridge regression with cross-validation
def cv(voxel, database_train, alpha_values):
    df_train = database_train.get_voxel_values(voxel)
    X_train = df_train.drop(columns=["fmri_value"]).values
    y_train = df_train["fmri_value"].values  

    ridge = Ridge()
    param_grid = {'alpha': alpha_values}
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    return voxel, best_alpha

# Main function
def main():
    start_time = time.time()

    args = parse_arguments()
    img_size = tuple(args.img_size)
    mode = args.mode
    alpha_values = args.alpha_values
    num_jobs = args.num_jobs  # Get the number of parallel jobs from command line
    print(f"Running with image size: {img_size}, mode: {mode}")
    print(f"Alpha values: {alpha_values}")
    print(f"Using {num_jobs} parallel jobs for computation.")

    paths = get_paths()
    database_train = load_dataset(args, paths)

    voxel_list = list(np.ndindex(img_size))
    top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
    top_voxels = get_top_voxels(database_train, img_size, voxel_list, top_voxels_path)
    print(f"Using {len(top_voxels)} top voxels for analysis.")

    results = Parallel(n_jobs=num_jobs)(
        delayed(cv)(voxel, database_train, alpha_values) for voxel in top_voxels
    )

    best_alphas = [result[1] for result in results]
    most_common_alpha = Counter(best_alphas).most_common(1)[0][0]
    print(f"Most common best alpha across top voxels: {most_common_alpha}")

    # Get unique filename to avoid overwriting
    unique_results_path = get_unique_filename(paths["results_path"], "best_alphas.csv")

    df_results = pd.DataFrame(results, columns=["Voxel", "Best_Alpha"])
    df_results.to_csv(unique_results_path, index=False)
    print(f"Best alpha values saved to {unique_results_path}!")

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

# Run the script
if __name__ == "__main__":
    main()