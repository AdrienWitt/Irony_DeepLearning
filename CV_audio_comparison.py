import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_helpers


os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'


# args = argparse.Namespace(
#     use_audio = True,
#     use_text = False,
#     use_base_features=True,
#     use_text_weighted = False,
#     use_audio_opensmile = True,
#     include_tasks = ["irony", "sarcasm"],
#     use_pca=True, num_jobs = 1, alpha = 0.1, pca_threshold = 0.5, use_umap = False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare audio features and openSMILE audio features for fMRI prediction.")
    
    # Dataset-related arguments
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                             help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--alpha", type=float, default=1.0,
                             help="Alpha value for Ridge regression (default: 1.0)")
    dataset_group.add_argument("--use_pca", action="store_true",
                             help="Use PCA for dimensionality reduction (default: False)")
    dataset_group.add_argument("--use_umap", action="store_true",
                             help="Use UMAP for dimensionality reduction (default: False)")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.80,
                             help="PCA threshold for dataset (default: 0.80)")
    dataset_group.add_argument("--num_jobs", type=int, default=4,
                             help="Number of parallel jobs (default: 4)")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony"],
                             help="List of tasks to include (default: sarcasm, irony)")

    return parser.parse_args()


def compare_audio_features(voxel, df_train, alpha):
    """Compare different audio feature types for a given voxel using 5-fold CV."""
    
    # Create a unique random seed from voxel coordinates
    random_seed = int(voxel[0] * 10000 + voxel[1] * 100 + voxel[2])
    
    # Identify feature columns
    wav2vec_cols = [col for col in df_train.columns if col.startswith(('emb_audio_', 'pc_audio_')) and not col.startswith(('emb_audio_opensmile_', 'pc_audio_opensmile_'))]
    opensmile_cols = [col for col in df_train.columns if col.startswith(('emb_audio_opensmile_', 'pc_audio_opensmile_'))]
    base_cols = [col for col in df_train.columns if not col.startswith(('emb_', 'pc_')) and col != "fmri_value"]
    
    # Extract target and features
    y = df_train["fmri_value"].values
    
    # Filter out background voxels (zero values)
    valid_idx = y != 0
    if sum(valid_idx) < len(y) * 0.5:  # If too few valid samples
        print(f"Not enough data for voxel {voxel} after filtering")
        return voxel, {
            'wav2vec': 0,
            'opensmile': 0,
            'combined': 0,
            'wav2vec_with_base': 0,
            'opensmile_with_base': 0,
            'combined_with_base': 0,
            'n_samples': 0
        }
    
    # Prepare datasets
    X_wav2vec = df_train[wav2vec_cols].values[valid_idx]
    X_opensmile = df_train[opensmile_cols].values[valid_idx]
    
    if len(base_cols) > 0:
        X_base = df_train[base_cols].values[valid_idx]
        X_wav2vec_with_base = np.hstack([X_wav2vec, X_base])
        X_opensmile_with_base = np.hstack([X_opensmile, X_base])
        X_combined_with_base = np.hstack([X_wav2vec, X_opensmile, X_base])
    else:
        X_wav2vec_with_base = X_wav2vec
        X_opensmile_with_base = X_opensmile
        X_combined_with_base = np.hstack([X_wav2vec, X_opensmile])
    
    X_combined = np.hstack([X_wav2vec, X_opensmile])
    y_filtered = y[valid_idx]
    
    # Setup cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    
    # Define models
    models = {
        'wav2vec': Ridge(alpha=alpha, random_state=random_seed),
        'opensmile': Ridge(alpha=alpha, random_state=random_seed),
        'combined': Ridge(alpha=alpha, random_state=random_seed),
        'wav2vec_with_base': Ridge(alpha=alpha, random_state=random_seed),
        'opensmile_with_base': Ridge(alpha=alpha, random_state=random_seed),
        'combined_with_base': Ridge(alpha=alpha, random_state=random_seed)
    }
    
    # Prepare datasets
    datasets = {
        'wav2vec': X_wav2vec,
        'opensmile': X_opensmile,
        'combined': X_combined,
        'wav2vec_with_base': X_wav2vec_with_base,
        'opensmile_with_base': X_opensmile_with_base,
        'combined_with_base': X_combined_with_base
    }
    
    # Compute scores
    scores = {}
    for name, model in models.items():
        if datasets[name].shape[1] > 0:  # Only if we have features
            scores[name] = np.mean(cross_val_score(model, datasets[name], y_filtered, cv=cv, scoring='r2'))
        else:
            scores[name] = 0
    
    scores['n_samples'] = sum(valid_idx)
    
    print(f"Voxel {voxel}: wav2vec={scores['wav2vec']:.4f}, opensmile={scores['opensmile']:.4f}, combined={scores['combined']:.4f}")
    
    return voxel, scores


def process_voxel(voxel, df_train, alpha):
    """Process a single voxel."""
    return compare_audio_features(voxel, df_train, alpha)


def main():
    start_time = time.time()
    args = parse_arguments()
    
    # Print configuration
    print(f"\nRunning audio feature comparison with configuration:")
    print(f"- Use base features: {args.use_base_features}")
    print(f"- Use PCA: {args.use_pca}")
    print(f"- Use UMAP: {args.use_umap}")
    print(f"- PCA threshold: {args.pca_threshold}")
    print(f"- Alpha value: {args.alpha}")
    print(f"- Number of parallel jobs: {args.num_jobs}")
    print(f"- Included tasks: {', '.join(args.include_tasks)}")
    
    # Load paths
    paths = analysis_helpers.get_paths()
    
    # Prepare dataset arguments
    dataset_args = {
        "use_base_features": args.use_base_features,
        "use_text": False,
        "use_audio": True,  # We need wav2vec features
        "use_audio_opensmile": True,  # We need openSMILE features
        "use_text_weighted": False,
        "use_pca": args.use_pca,
        "use_umap": args.use_umap,
        "pca_threshold": args.pca_threshold,
        "include_tasks": args.include_tasks
    }
    
    # Load dataset
    participant_list = os.listdir(paths["data_path"])
    #participant_list = participant_list[:10]  # Limit to first 10 participants for faster processing
    print(f"\nLoading dataset for {len(participant_list)} participants...")
    
    # Set args namespace for dataset loading
    args_namespace = argparse.Namespace(**dataset_args)
    database = analysis_helpers.load_dataset(args_namespace, paths, participant_list)
    
    # Load or compute top voxels
    img_size = (79, 95, 79)
    voxel_list = list(np.ndindex(img_size))
    top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
    top_voxels = analysis_helpers.get_top_voxels(database, img_size, voxel_list, top_voxels_path)
    print(f"\nUsing {len(top_voxels)} top voxels for analysis.")
    
    # Configure parallel processing
    n_jobs = args.num_jobs if args.num_jobs > 0 else os.cpu_count()
    
    # Use Joblib's Parallel and delayed with error handling
    try:
        print(f"\nRunning parallel processing with {n_jobs} jobs...")
        results_with_voxels = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_voxel)(voxel, database.get_voxel_values(voxel), args.alpha) for voxel in top_voxels
        )
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        print("Falling back to sequential processing...")
        # Fallback to sequential processing
        results_with_voxels = []
        for voxel in top_voxels:
            try:
                result = process_voxel(voxel, database, args.alpha)
                results_with_voxels.append(result)
            except Exception as e:
                print(f"Error processing voxel {voxel}: {str(e)}")
    
    # Unpack results
    results = {voxel: result for voxel, result in results_with_voxels}
    
    # Calculate average scores for each method
    all_scores = {
        'wav2vec': [],
        'opensmile': [],
        'combined': [],
        'wav2vec_with_base': [],
        'opensmile_with_base': [],
        'combined_with_base': []
    }
    
    voxel_counts = {'total': 0, 'wav2vec_better': 0, 'opensmile_better': 0, 'combined_better': 0}
    
    for scores in results.values():
        if scores['n_samples'] > 0:  # Only count voxels with valid data
            voxel_counts['total'] += 1
            for key in all_scores.keys():
                all_scores[key].append(scores[key])
            
            # Count which method is best for this voxel
            wav2vec_score = scores['wav2vec']
            opensmile_score = scores['opensmile']
            combined_score = scores['combined']
            
            if wav2vec_score > opensmile_score and wav2vec_score > combined_score:
                voxel_counts['wav2vec_better'] += 1
            elif opensmile_score > wav2vec_score and opensmile_score > combined_score:
                voxel_counts['opensmile_better'] += 1
            elif combined_score > 0:  # Only count if combined score is valid
                voxel_counts['combined_better'] += 1
    
    # Compute mean scores
    mean_scores = {key: np.mean(scores) if scores else 0 for key, scores in all_scores.items()}
    
    # Print results
    print("\n=== Audio Feature Comparison Results ===")
    print(f"Total voxels analyzed: {voxel_counts['total']}")
    print("\nMean R² Scores:")
    print(f"- Wav2Vec features: {mean_scores['wav2vec']:.4f}")
    print(f"- OpenSMILE features: {mean_scores['opensmile']:.4f}")
    print(f"- Combined features: {mean_scores['combined']:.4f}")
    if args.use_base_features:
        print(f"- Wav2Vec + base features: {mean_scores['wav2vec_with_base']:.4f}")
        print(f"- OpenSMILE + base features: {mean_scores['opensmile_with_base']:.4f}")
        print(f"- Combined + base features: {mean_scores['combined_with_base']:.4f}")
    
    print("\nVoxel Counts by Best Predictor:")
    print(f"- Wav2Vec better: {voxel_counts['wav2vec_better']} ({voxel_counts['wav2vec_better']/voxel_counts['total']*100:.1f}%)")
    print(f"- OpenSMILE better: {voxel_counts['opensmile_better']} ({voxel_counts['opensmile_better']/voxel_counts['total']*100:.1f}%)")
    print(f"- Combined better: {voxel_counts['combined_better']} ({voxel_counts['combined_better']/voxel_counts['total']*100:.1f}%)")
    
    # Create and save a DataFrame with results
    df_results = pd.DataFrame([
        {
            'voxel_x': voxel[0],
            'voxel_y': voxel[1],
            'voxel_z': voxel[2],
            'wav2vec': scores['wav2vec'],
            'opensmile': scores['opensmile'],
            'combined': scores['combined'],
            'wav2vec_with_base': scores['wav2vec_with_base'],
            'opensmile_with_base': scores['opensmile_with_base'],
            'combined_with_base': scores['combined_with_base'],
            'n_samples': scores['n_samples'],
            'best_predictor': 'wav2vec' if scores['wav2vec'] > scores['opensmile'] and scores['wav2vec'] > scores['combined'] else
                             'opensmile' if scores['opensmile'] > scores['wav2vec'] and scores['opensmile'] > scores['combined'] else
                             'combined'
        }
        for voxel, scores in results.items() if scores['n_samples'] > 0
    ])
    
    # Save results
    task_code = "_".join([task[:3] for task in args.include_tasks])
    pca_str = f"_pca{args.pca_threshold}" if args.use_pca else ""
    base_str = "_withbase" if args.use_base_features else ""
    results_path = os.path.join(paths["results_path"], f"audio_comparison{pca_str}{base_str}_{task_code}.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to {results_path}")
    
    # Create a visualization
    try:
        plt.figure(figsize=(10, 6))
        comparison_data = pd.melt(df_results[['voxel_x', 'voxel_y', 'voxel_z', 'wav2vec', 'opensmile', 'combined']], 
                                  id_vars=['voxel_x', 'voxel_y', 'voxel_z'], 
                                  value_vars=['wav2vec', 'opensmile', 'combined'],
                                  var_name='Feature Type', value_name='R² Score')
        
        sns.boxplot(x='Feature Type', y='R² Score', data=comparison_data)
        plt.title('Comparison of Audio Feature Types for fMRI Prediction')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(paths["results_path"], f"audio_comparison{pca_str}{base_str}_{task_code}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {plot_path}")
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main() 