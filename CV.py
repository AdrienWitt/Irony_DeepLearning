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
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer


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
    dataset_group.add_argument("--use_pca", action="store_true", 
                               help="Use PCA for embeddings with the a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, nargs='+', default=[0.70, 0.80, 0.90],
                           help="List of explained variance thresholds for text PCA (default: [0.90, 0.95, 0.99]).")

    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--alpha_values", type=float, nargs='+', default=[1, 10, 100],
                        help="List of alpha values for Ridge regression (default: [1, 10, 100]).")
    analysis_group.add_argument("--num_jobs", type=int, default=20,
                                help="Number of parallel jobs for voxel processing (default: -1 for all cores).")
    analysis_group.add_argument("--optimize_pca_threshold", action="store_true",
                                help="Optimize PCA threshold using cross-validation (default: False).")

    return parser.parse_args()

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

def get_unique_filename(base_path, filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(base_path, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return os.path.join(base_path, new_filename)

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
        "pca_threshold": args.pca_threshold,
        "use_pca" : args.use_pca
    }

    database_train = dataset.BaseDataset(participant_list=train_participants, **dataset_args)

    return database_train

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

def cv(voxel, database_train, alpha_values, pca_thresholds):
    df_train = database_train.get_voxel_values(voxel)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = df_train.drop(columns=["fmri_value"])
    y_train = df_train["fmri_value"]  

    # Identify which columns correspond to text and audio
    text_cols = [col for col in X_train.columns if col.startswith("emb_text_")]
    audio_cols = [col for col in X_train.columns if col.startswith('emb_audio_')]

    # Define the pipeline with PCA and Ridge regression
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('text', PCA(), text_cols),  # Apply PCA to text columns
                ('audio', PCA(), audio_cols),  # Apply PCA to audio columns
            ],
            remainder='passthrough'  # Keep non-text, non-audio columns unchanged
        )),
        ('ridge', Ridge())
    ])

    # Define the parameter grid for PCA threshold (same threshold for both text and audio) and Ridge alpha
    param_grid = {
        'preprocessor__text__n_components': pca_thresholds,  # Same PCA threshold for text
        'preprocessor__audio__n_components': pca_thresholds,  # Same PCA threshold for audio
        'ridge__alpha': alpha_values  # Optimize Ridge alpha
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Return the best PCA components for both text and audio and the best alpha value
    return grid_search.best_params_['preprocessor__text__n_components'], \
           grid_search.best_params_['preprocessor__audio__n_components'], \
           grid_search.best_params_['ridge__alpha']

def main():
    start_time = time.time()

    args = parse_arguments()
    img_size = tuple(args.img_size)
    mode = args.mode
    alpha_values = args.alpha_values
    pca_thresholds = args.pca_threshold
    num_jobs = args.num_jobs  # Get the number of parallel jobs from command line
    print(f"Running with settings:\n"
    f"- Image size: {args.img_size}\n"
    f"- Mode: {args.mode}\n"
    f"- Use base features: {args.use_base_features}\n"
    f"- PCA thresholds: {args.pca_threshold}\n"
    f"- Ridge alpha: {args.alpha_values}\n"
    f"- Number of parallel jobs: {args.num_jobs}")

    paths = get_paths()
    database_train = load_dataset(args, paths)

    voxel_list = list(np.ndindex(img_size))
    top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
    top_voxels = get_top_voxels(database_train, img_size, voxel_list, top_voxels_path)
    print(f"Using {len(top_voxels)} top voxels for analysis.")

    results = Parallel(n_jobs=num_jobs)(
        delayed(cv)(voxel, database_train, alpha_values, pca_thresholds) for voxel in top_voxels
    )

    best_pca_thresholds = [result[0] for result in results]  # These are PCA components
    best_alphas = [result[1] for result in results]
    
    most_common_pca_threshold = Counter(best_pca_thresholds).most_common(1)[0][0]
    most_common_alpha = Counter(best_alphas).most_common(1)[0][0]
    print(f"Most common best PCA threshold across top voxels: {most_common_pca_threshold}")
    print(f"Most common best alpha across top voxels: {most_common_alpha}")

    # Get unique filename to avoid overwriting
    unique_results_path = get_unique_filename(paths["results_path"], "best_params.csv")

    df_results = pd.DataFrame(results, columns=["Voxel", "Best_PCA_Threshold", "Best_Alpha"])
    df_results.to_csv(unique_results_path, index=False)
    print(f"Best parameters saved to {unique_results_path}!")

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()