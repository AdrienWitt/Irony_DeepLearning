import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import analysis_helpers


os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

args = argparse.Namespace(
    use_audio = False,
    use_text = False,
    use_base_features=True,
    use_text_weighted = True,
    use_audio_opensmile = True,
    include_tasks = ["irony", "sarcasm"],
    use_pca=False, num_jobs = 1, alpha = 0.1, pca_thresholds = [0.5, 0.6], 
    use_umap = False, step = 2, step2_use_best_base = False, pca_threshold = 0.5)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # **Analysis step selection**
    step_group = parser.add_argument_group("Analysis Step Selection")
    step_group.add_argument("--step", type=int, choices=[1, 2, 3], required=True,
                          help="Analysis step to run: 1=base features comparison, 2=PCA threshold optimization, 3=alpha optimization")
    
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
    dataset_group.add_argument("--use_audio_opensmile", action="store_true", 
                                help="Include openSMILE audio features in dataset (default: False).")
    dataset_group.add_argument("--use_pca", action="store_true",
                             help="Use PCA for embeddings (default: False)")
    dataset_group.add_argument("--use_umap", action="store_true",
                             help="Use umaps for embeddings (default: False)")
    dataset_group.add_argument("--pca_threshold", type=float, default=1,
                             help="PCA threshold for dataset (default: 0.50)")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony", "prosody", "semantic", "tom"],
                            help="List of tasks to include (default: all available tasks).")
    
    # **Analysis-related arguments**
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--num_jobs", type=int, default=20,
                              help="Number of parallel jobs for voxel processing (default: 20).")
    
    # Step-specific arguments
    step1_group = parser.add_argument_group("Step 1: Base Features Comparison")
    step1_group.add_argument("--fixed_alpha", type=float, default=1,
                           help="Fixed alpha value for base features comparison (default: 100.0)")
    
    step2_group = parser.add_argument_group("Step 2: PCA Threshold Optimization")
    step2_group.add_argument("--step2_use_best_base", action="store_true",
                           help="Use the best base features configuration from step 1 (default: False)")
    step2_group.add_argument("--pca_thresholds", type=float, nargs='+', default=[0.50, 0.60, 0.70, 0.80],
                            help="List of PCA thresholds for optimization (default: [0.50, 0.60, 0.70, 0.80])")
    
    step3_group = parser.add_argument_group("Step 3: Alpha Optimization")
    step3_group.add_argument("--alpha_values", type=float, nargs='+', default=[0.1, 0.5, 1, 5],
                           help="List of alpha values for Ridge regression (default: [0.1, 0.5, 1, 5])")
    step3_group.add_argument("--step3_use_best_pca", action="store_true",
                           help="Use the best PCA threshold from step 2 (default: False)")
    step3_group.add_argument("--step3_use_best_base", action="store_true",
                           help="Use the best base features configuration from step 1 (default: False)")

    args = parser.parse_args()
        
    return args

def cv(df_train, voxel, alpha_values, pca_thresholds, step, fixed_alpha=None, use_best_base=False, use_best_pca=False):
    paths = analysis_helpers.get_paths()
        
    # Create a unique random seed from voxel coordinates
    random_seed = int(voxel[0] * 10000 + voxel[1] * 100 + voxel[2])
    
    df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    
    # Identify embedding columns
    text_cols = [col for col in df_train.columns if col.startswith(('emb_weighted_', 'pc_weighted_'))]
    audio_cols = [col for col in df_train.columns if col.startswith(('emb_audio_', 'pc_audio_'))]
    print(audio_cols)
    embedding_cols = text_cols + audio_cols    
    
    # Prepare features and target
    X_train = df_train.drop(columns=["fmri_value"])
    y_train = df_train["fmri_value"]  # Keep as pandas Series like in working code

    valid_idx = y_train != 0 ## exclude background values
    X_filtered = X_train[valid_idx]
    y_filtered = y_train[valid_idx]
    
    print(f"Filtered data contains {len(X_filtered)} valid entries.")

    if step == 1:  # Base features comparison with fixed alpha
        # Create two different feature sets
        X_with_base = X_filtered
        X_without_base = X_filtered[embedding_cols]  # Only keep embedding columns
        
        # Define pipelines for both configurations without PCA
        pipeline_with_base = Pipeline([
            ('ridge', Ridge(alpha=fixed_alpha, random_state=random_seed))
        ])
        
        pipeline_without_base = Pipeline([
            ('ridge', Ridge(alpha=fixed_alpha, random_state=random_seed))
        ])
        
        # Fit both pipelines
        pipeline_with_base.fit(X_with_base, y_filtered)
        pipeline_without_base.fit(X_without_base, y_filtered)
        
        # Get scores
        with_base_score = pipeline_with_base.score(X_with_base, y_filtered)
        without_base_score = pipeline_without_base.score(X_without_base, y_filtered)
        
        return {
            'with_base': {'score': with_base_score},
            'without_base': {'score': without_base_score}
        }
    
    elif step == 2:  # PCA threshold optimization
        # Determine whether to use base features based on previous results
        if use_best_base:
            # Load previous results to determine best configuration
            prev_results_path = os.path.join(paths["results_path"], "step1_results.csv")
            prev_results = pd.read_csv(prev_results_path)
            use_base = prev_results['with_base_score'].mean() > prev_results['without_base_score'].mean()
        else:
            use_base = True  # Default to using base features
        
        # Select appropriate feature set
        X = X_filtered if use_base else X_filtered[embedding_cols]
        
        # Optimize PCA computation by pre-filtering the columns
        text_features = X[text_cols]
        audio_features = X[audio_cols]
        other_features = X.drop(columns=text_cols + audio_cols)
        
        # Pre-compute base PCA with maximum threshold to avoid recomputing for each threshold
        print("Pre-computing PCA components...")
        max_threshold = max(pca_thresholds)
        text_pca = PCA(n_components=max_threshold)
        audio_pca = PCA(n_components=max_threshold)
        
        # Fit PCAs once
        text_transformed_full = text_pca.fit_transform(text_features)
        audio_transformed_full = audio_pca.fit_transform(audio_features)
        
        print(f"Text PCA: {text_transformed_full.shape[1]} components explain {max_threshold*100:.1f}% variance")
        print(f"Audio PCA: {audio_transformed_full.shape[1]} components explain {max_threshold*100:.1f}% variance")
        
        # Get explained variance ratios
        text_exp_var = text_pca.explained_variance_ratio_.cumsum()
        audio_exp_var = audio_pca.explained_variance_ratio_.cumsum()
        
        # Create simplified custom transformer that just selects components
        class OptimizedPCA:
            def __init__(self, n_components=0.5):
                self.n_components = n_components
                self.text_components = None
                self.audio_components = None
                
            def fit(self, X, y=None):
                # Determine number of components to use based on threshold
                self.text_components = np.sum(text_exp_var <= self.n_components) + 1
                self.audio_components = np.sum(audio_exp_var <= self.n_components) + 1
                self.text_components = min(self.text_components, text_transformed_full.shape[1])
                self.audio_components = min(self.audio_components, audio_transformed_full.shape[1])
                return self
                
            def transform(self, X):
                # For cross-validation, we need to apply the fitted PCA to the current subset
                # Get the current subset of features
                text_features_subset = X[text_cols]
                audio_features_subset = X[audio_cols]
                other_feat_subset = X.drop(columns=text_cols + audio_cols)
                
                # Apply the pre-fitted PCA transformations but only keep the number of components we determined
                text_transformed_subset = text_pca.transform(text_features_subset)[:, :self.text_components]
                audio_transformed_subset = audio_pca.transform(audio_features_subset)[:, :self.audio_components]
                
                # Combine the transformed features with the other features
                result = np.hstack([text_transformed_subset, audio_transformed_subset, other_feat_subset])
                return result
                
            def get_params(self, deep=True):
                return {"n_components": self.n_components}
                
            def set_params(self, **params):
                self.n_components = params["n_components"]
                return self
        
        pipeline = Pipeline([
            ('preprocessor', OptimizedPCA()),
            ('ridge', Ridge(alpha=fixed_alpha, random_state=random_seed))
        ])
        
        # Now we only need to search over one parameter
        param_grid = {
            'preprocessor__n_components': pca_thresholds
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', verbose=1)
        grid_search.fit(X, y_filtered)
        
        best_threshold = grid_search.best_params_['preprocessor__n_components']
        return {
            'best_pca_text': best_threshold,
            'best_pca_audio': best_threshold,
            'score': grid_search.best_score_
        }
    
    else:  # step 3: Alpha optimization
        # Determine whether to use base features and best PCA threshold
        if use_best_base:
            prev_results_path = os.path.join(paths["results_path"], "step1_results.csv")
            prev_results = pd.read_csv(prev_results_path)
            use_base = prev_results['with_base_score'].mean() > prev_results['without_base_score'].mean()
        else:
            use_base = True
               
        # Select appropriate feature set
        X = X_filtered if use_base else X_filtered[embedding_cols]        
        # Define pipeline without PCA (since it's already done in the dataset)
        pipeline = Pipeline([
            ('ridge', Ridge(random_state=random_seed))
        ])
        
        param_grid = {'ridge__alpha': alpha_values}
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
        grid_search.fit(X, y_filtered)
        
        return {
            'best_alpha': grid_search.best_params_['ridge__alpha'],
            'score': grid_search.best_score_
        }

def process_voxel(voxel, df_train, args):
    """Process a single voxel using the appropriate step"""
    
    if args.step == 1:
        return voxel, cv(df_train, voxel, None, None, args.step, fixed_alpha=args.fixed_alpha)
    elif args.step == 2:
        return voxel, cv(df_train, voxel, None, args.pca_thresholds, args.step, 
                      fixed_alpha=args.fixed_alpha, use_best_base=args.step2_use_best_base)
    else:  # step 3
        return voxel, cv(df_train, voxel, args.alpha_values, None, args.step, 
                      use_best_base=args.step3_use_best_base, use_best_pca=args.step3_use_best_pca)
    
def main():
    start_time = time.time()
    args = parse_arguments()
    
    # Print configuration
    print(f"\nRunning Step {args.step} with configuration:")
    print(f"- Use base features: {args.use_base_features}")
    print(f"- Use text: {args.use_text}")
    print(f"- Use audio: {args.use_audio}")
    print(f"- Use audio opensmile: {args.use_audio_opensmile}")
    print(f"- Use text_weighted: {args.use_text_weighted}")
    print(f"- Use PCA: {args.use_pca}")
    print(f"- Include tasks: {args.include_tasks}")
    print(f"- Number of parallel jobs: {args.num_jobs}")
    
    if args.step == 1:
        print(f"- Fixed alpha: {args.fixed_alpha}")
    elif args.step == 2:
        print(f"- PCA thresholds: {args.pca_thresholds}")
        print(f"- Use best base features config: {args.step2_use_best_base}")
    else:  # step 3
        print(f"- Alpha values: {args.alpha_values}")
        print(f"- Use best base features config: {args.step3_use_best_base}")
        print(f"- Use best PCA threshold: {args.step3_use_best_pca}")

    paths = analysis_helpers.get_paths()
    img_size = (79, 95, 79)
    
    # Get list of voxels to process
    voxel_list = list(np.ndindex(tuple(img_size)))
    
    # Load dataset once
    participant_list = os.listdir(paths["data_path"])
    #participant_list = participant_list[:10]
    database_train = analysis_helpers.load_dataset(args, paths, participant_list)
    
    # Get top voxels
    top_voxels_path = os.path.join(paths["results_path"], "top10_voxels.csv")
    top_voxels = analysis_helpers.get_top_voxels(database_train, tuple(img_size), voxel_list, top_voxels_path)
    #top_voxels = top_voxels[:100]
    print(f"\nUsing {len(top_voxels)} top voxels for analysis.")
    
    # Configure parallel processing
    n_jobs = args.num_jobs if args.num_jobs > 0 else os.cpu_count()
    backend = 'loky'  # Use loky backend which is more robust
    
    # Use Joblib's Parallel and delayed with error handling
    try:
        results_with_voxels = Parallel(n_jobs=n_jobs, backend=backend, verbose=1)(
            delayed(process_voxel)(voxel, database_train.get_voxel_values(voxel), args) for voxel in top_voxels
        )
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        print("Falling back to sequential processing...")
        # Fallback to sequential processing
        results_with_voxels = []
        for voxel in top_voxels:
            try:
                result = process_voxel(voxel, database_train.get_voxel_values(voxel), args)
                results_with_voxels.append(result)
            except Exception as e:
                print(f"Error processing voxel {voxel}: {str(e)}")
                continue
    
    # Unpack results
    results = {voxel: result for voxel, result in results_with_voxels}

    # Process and save results
    if args.step == 1:
        # Calculate average scores
        with_base_scores = [result['with_base']['score'] for result in results.values()]
        without_base_scores = [result['without_base']['score'] for result in results.values()]
        
        avg_with_base = np.mean(with_base_scores)
        avg_without_base = np.mean(without_base_scores)
        
        print("\nBase Features Comparison Results:")
        print(f"Average R² score with base features: {avg_with_base:.4f}")
        print(f"Average R² score without base features: {avg_without_base:.4f}")
        print(f"Difference: {avg_with_base - avg_without_base:.4f}")
        
        # Save results
        df_results = pd.DataFrame([
            {
                'voxel': voxel,
                'with_base_score': result['with_base']['score'],
                'without_base_score': result['without_base']['score']
            }
            for voxel, result in results.items()
        ])
        results_path = os.path.join(paths["results_path"], "step1_results.csv")
    
    elif args.step == 2:
        # Calculate average scores for each PCA threshold
        best_pca_texts = [result['best_pca_text'] for result in results.values()]
        best_pca_audios = [result['best_pca_audio'] for result in results.values()]
        scores = [result['score'] for result in results.values()]
        
        print("\nPCA Threshold Optimization Results:")
        print(f"Average R² score: {np.mean(scores):.4f}")
        print(f"Most common best PCA threshold for text: {Counter(best_pca_texts).most_common(1)[0][0]}")
        print(f"Most common best PCA threshold for audio: {Counter(best_pca_audios).most_common(1)[0][0]}")
        
        # Save results
        df_results = pd.DataFrame([
            {
                'voxel': voxel,
                'best_pca_text': result['best_pca_text'],
                'best_pca_audio': result['best_pca_audio'],
                'score': result['score']
            }
            for voxel, result in results.items()
        ])
        results_path = os.path.join(paths["results_path"], "step2_results.csv")
    
    else:  # step 3
        # Calculate average scores for each alpha
        best_alphas = [result['best_alpha'] for result in results.values()]
        scores = [result['score'] for result in results.values()]
        
        print("\nAlpha Optimization Results:")
        print(f"Average R² score: {np.mean(scores):.4f}")
        print(f"Most common best alpha: {Counter(best_alphas).most_common(1)[0][0]}")
        
        # Save results
        df_results = pd.DataFrame([
            {
                'voxel': voxel,
                'best_alpha': result['best_alpha'],
                'score': result['score']
            }
            for voxel, result in results.items()
        ])
        results_path = os.path.join(paths["results_path"], "step3_results.csv")
    
    df_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}!")

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()